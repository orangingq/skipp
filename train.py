import torch
from models.pipeline import skipp_ScheduleGPipe
from utils import Metric, MetricGroup, args
import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, Schedule1F1B
from utils.util import get_first_stage_rank, get_last_stage_rank, log_time, pipeline_log

def validation(dataloader, model, criterion):
    metrics = MetricGroup([Metric('valid_avg_loss', 0.0), Metric('valid_avg_acc1', 0.0), Metric('valid_avg_acc5', 0.0)])
    model.eval()
    with torch.no_grad():
        step = 0
        batch_iterator = iter(dataloader)
        num_batches = len(dataloader)
        schedule = skipp_ScheduleGPipe(model, args.microbatches, log_pp=False) # for pipeline parallelism

        while step < num_batches:
            step += 1
            batch = next(batch_iterator)
            inputs, targets = batch[0].to(get_first_stage_rank()), batch[1].to(get_last_stage_rank())
            
            if args.pp == 1:
                logits = model(inputs)
            else: # pipeline parallelism
                assert schedule is not None, "schedule should be provided for pipeline parallelism"
                logits = schedule.step(inputs)
                
            if args.local_rank != get_last_stage_rank():
                continue
            
            # compute loss and accuracy
            loss = criterion(logits, targets)
            _, top5_preds = logits.topk(5, 1, largest=True, sorted=True)
            top5_preds = top5_preds.t()
            top1_preds = top5_preds[0]
            top1_acc = torch.sum(top1_preds == targets)/targets.size(0)
            top5_acc = torch.sum(torch.sum(top5_preds == targets, dim=0, dtype=torch.bool))/targets.size(0)
            metrics.step([loss.detach(), top1_acc, top5_acc])
    return metrics.avg


def train(dataloader, model, criterion, optimizer, regularization=None):
    metrics = MetricGroup([Metric('train_avg_loss', 0.0), Metric('train_avg_acc', 0.0)]) # trace loss/acc of every batch
    metrics_log = MetricGroup([Metric('train_log_loss', 0.0), Metric('train_log_acc', 0.0)]) # trace loss/acc for each log_freq

    model.train()
    step = 0
    batch_iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    
    if args.pp_scheduler == 'gpipe':
        schedule = skipp_ScheduleGPipe(model, args.microbatches, criterion, log_pp=model.get_log_pp())
    else:
        schedule = Schedule1F1B(model, args.microbatches, criterion, log_pp=model.get_log_pp())

    while step < args.num_batches:
        step += 1
        batch = next(batch_iterator)
        inputs, targets = batch[0].to(get_first_stage_rank()), batch[1].to(get_last_stage_rank())
        
        if step >= 4 and model.get_log_pp(): # log the pipeline flows of first 3 steps of the first epoch
            model.set_log_pp(False)
            schedule.set_log_pp(False)

        if model.get_log_pp() and args.local_rank == get_first_stage_rank():
            pipeline_log(microbatch=0, step='start', postfix=f"Input : {inputs.shape}")
            
        if args.pp == 1:
            logits = model(inputs)
            
            # compute loss
            loss = criterion(logits, targets)
            if regularization is not None: # regularization term
                reg = regularization(model)
                loss = loss + args.lamb * reg

            # update model
            loss.backward()
            _, preds = logits.max(1)

        else: # pipeline parallelism
            losses = [0] * args.microbatches
            assert schedule is not None, "schedule should be provided for pipeline parallelism"
            # forward and backward
            schedule.step(inputs, target=targets, losses=losses)

            if args.local_rank == get_last_stage_rank(): # get the predictions and loss
                _, preds = torch.concat(model.output_chunks).max(1)
                loss = torch.stack(losses).mean() # average the losses of microbatches
        
        # statistics
        if args.local_rank == get_last_stage_rank():
            metrics.step([loss.detach(), torch.sum(preds == targets.data)/targets.size(0)])
            metrics_log.step([loss.detach(), torch.sum(preds == targets.data)/targets.size(0)])
        
        # dist.barrier() # TODO ? erase?
        if model.get_log_pp() and args.local_rank == get_last_stage_rank():
            pipeline_log(microbatch=0, step='finish', postfix=f"Loss : {loss.item()}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to reproduce
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Log during an epoch
        if args.log_freq is not None and step % args.log_freq == 0:
            log_time(f"\tStep {step}/{args.num_batches}: {metrics_log}")
            metrics_log.reset()
    
    return metrics.avg
    