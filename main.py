import torch
import sys
import time
import torch.distributed as dist
import wandb
from checkpoints.save import load_checkpoint, save_checkpoint
from utils import args
from utils import random_seed, TimeMetric, set_args, get_args, get_default_path, FLOP_counter, get_last_stage_rank
from models import get_model
from optimizers.optimizer import get_optimizer_scheduler
from train import train, validation
from utils.util import log_time
sys.path.append(get_default_path('dataset'))
from dataloader import get_dataloaders

def main():    
    set_args() # set the arguments
    random_seed(args.seed)

    # 1) Dataset Load
    num_workers = 4 if args.pp==1 else 0
    dataloaders, _, args.num_classes, args.image_size = get_dataloaders(args.dataset, batch_size=args.batch_size, num_workers=num_workers, image_size=224, 
                                                                        shuffle=True, distributed=False, autoaugment=False)
    
    args.num_batches = len(dataloaders['train'])
    if args.dp > 1:
        dist.barrier()

    # 2) Model Load, Loss Function, Optimizer, Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    batch_shape = next(iter(dataloaders['train']))[0].shape
    model = get_model(args.model_name, batch_shape)
    optimizer, lr_scheduler, _ = get_optimizer_scheduler(model)

    # 4) Load Model Checkpoint
    start_epoch = 1
    if args.checkpoint is not None:
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, load_dir=args.checkpoint)
        start_epoch = last_epoch + 1

    # 5) Logger settings
    if args.wandb:
        args.wandb = True
        wandb.login()
        wandb.init(project ='Skipp', name = args.run_name, config=get_args())
    best_acc1, best_epoch1 = 0.0, 0
    best_acc5 = 0.0
    train_time = TimeMetric("Training Time", time.time())
    epoch_time = TimeMetric("Epoch Training Time", time.time())
    log = {}
    
    # 6) Train!
    for epoch in range(start_epoch, args.num_epochs+1):
        if args.local_rank == args.master_rank : 
            log_time(f"\nEpoch {epoch}/{args.num_epochs}\n"+"-"*10+"\n",end="", rank=False)
        epoch_time.reset() 

        # Train and Validation
        train_avg_loss, train_avg_acc = train(dataloaders['train'], model, criterion, optimizer)
        val_avg_loss, val_avg_acc1, val_avg_acc5 = validation(dataloaders['val'], model, criterion)
        best_acc1, best_acc5 = max(best_acc1, val_avg_acc1), max(best_acc5, val_avg_acc5)
        best_epoch1 = epoch if best_acc1 == val_avg_acc1 else best_epoch1
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Log
        if args.local_rank == args.master_rank :
            print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_avg_acc:.4f}, Val Loss: {val_avg_loss:.4f}, Val Top1 Acc: {val_avg_acc1:.4f}, Val Top5 Acc: {val_avg_acc5:.4f}, ", end='')
            print(f"Best Acc: {best_acc1:.4f}, Best Epoch: {best_epoch1} \n{epoch_time}")
        
        if args.pp_rank == get_last_stage_rank():
            # flops = FLOP_counter(model, next(iter(dataloaders['train']))[0].shape)
            params = sum(p.numel() for p in model.parameters())

            log = {"train_loss": train_avg_loss, 
                    "train_Top1_accuracy": train_avg_acc, 
                    "val_loss": val_avg_loss,
                    "val_Top1_accuracy": val_avg_acc1,
                    "val_Top5_accuracy": val_avg_acc5,
                    # space efficiency
                    "memory footprint (GB)": torch.cuda.memory_allocated() / 1e9,
                    "model parameters": params,
                    # time efficiency
                    "epoch time": epoch_time.elapsed(),
                    # "GFLOPs": flops / 1e9, 
                    # others
                    "learning rate": optimizer.param_groups[0]['lr'],
                }
            if args.wandb:
                wandb.log(data=log, step=epoch)

            # save model
            if args.save_dir is not None: 
                args.save_dir = save_checkpoint(model, optimizer, epoch, args.save_dir, logs=log, first_save=(epoch==start_epoch))

            # Early stopping
            stop_threshold = 10
            if args.early_stop and epoch - best_epoch1 > stop_threshold:
                if args.local_rank == args.master_rank : 
                    print("\nEarly Stopping ...\n")
                break
            
    
    # 7) Log after training
    if args.pp_rank == get_last_stage_rank():
        if args.local_rank == args.master_rank : 
            time_elapsed = train_time.elapsed()
            print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print(f"Best Validation Accuracy: {best_acc1}, Epoch: {best_epoch1}")
            
            if args.wandb:
                final_log = {
                    "Best Validation Top1 Accuracy": best_acc1,
                    "Best Validation Top5 Accuracy": best_acc5,
                    "Best Epoch": best_epoch1,
                    "Total Training Time": time_elapsed,
                }
                wandb.log(final_log)
    # if args.local_rank == args.master_rank : 
    #     print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    #     print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc1, best_epoch1))


    # if args.wandb:
    #     final_log = {
    #         "Best Validation Top1 Accuracy": best_acc1,
    #         "Best Validation Top5 Accuracy": best_acc5,
    #         "Best Epoch": best_epoch1,
    #         "Total Training Time": time_elapsed,
    #     }
    #     wandb.log(final_log)

    return



if __name__ == "__main__":
    main()
