# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
###CSH
from collections import defaultdict
from utils.util import pipeline_log
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from torch.distributed.pipelining.schedules import PipelineScheduleSingle
import torch.distributed as dist
from torch.profiler import record_function

class skipp_ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """
    def __init__(self, *args: Any, log_pp=False, **kwds: Any):
        super().__init__(*args, **kwds)
        self.log_pp = log_pp
    
    def set_log_pp(self, log_pp: bool):
        self.log_pp = log_pp

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for key, work in works.items():
                    work.wait()
                    if self.log_pp:
                        pipeline_log(key, "fwd_recv", "wait end")

                output = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()
            if self.log_pp:
                pipeline_log(i, "fwd_send")

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):

                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for key, work in works.items():
                    work.wait()
                    if self.log_pp:
                        pipeline_log(key, "bwd_recv", "wait end")
                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(i, loss=loss)
                
                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())
        
        if self.log_pp:
            pipeline_log(0, "update", "start")
        self._stage.initialize()
        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)
        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()
            if self.log_pp:
                pipeline_log(i, "bwd_send")

class skipp_Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """
    def __call__(self, log_pp=False, *args: Any, **kwds: Any) -> Any:
        super().__call__(*args, **kwds)
        self.log_pp = log_pp
        return self
    
    def set_log_pp(self, log_pp: bool):
        self.log_pp = log_pp

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0
        weight_stage_mb_index = 0

        # Warmup phase
        send_work = None
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if recv_work := dist.batch_isend_irecv(fwd_recvs).pop():
                recv_work.wait() #TODODODODODODODODDO wait 다음에 pipeline_log 호출하도록~~~~~~ 이거 수정하고 다시 돌려보기
                if self.log_pp:
                    pipeline_log(fwd_mb_index, "fwd_recv", "wait end")

            # Compute
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            if send_work:
                send_work.wait()

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = dist.batch_isend_irecv(fwd_sends).pop()
                if self.log_pp:
                    pipeline_log(fwd_mb_index, "fwd_send")
            # otherwise:
            #   The last foward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            if fuse_work := dist.batch_isend_irecv(fwd_sends + bwd_recvs).pop():
                fuse_work.wait()
                if self.log_pp:
                    pipeline_log(bwd_mb_index, "fwd_send")
                    pipeline_log(fwd_mb_index, "bwd_recv", "wait end")

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(bwd_mb_index, loss=loss)

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            if fuse_work := dist.batch_isend_irecv(bwd_sends + fwd_recvs).pop():
                fuse_work.wait()
                if self.log_pp:
                    pipeline_log(bwd_mb_index, "bwd_send")
                    pipeline_log(fwd_mb_index, "fwd_recv", "wait end")

            # Now do the fwd
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = dist.batch_isend_irecv(bwd_sends).pop()

        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            if recv_work := dist.batch_isend_irecv(bwd_recvs).pop():
                recv_work.wait()
                if self.log_pp:
                    pipeline_log(bwd_mb_index, "bwd_recv", "wait end")

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(bwd_mb_index, loss=loss)

            # Clear previous chunk's backward sends (hopefully they have well finished)
            if send_work:
                send_work.wait()

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = dist.batch_isend_irecv(bwd_sends).pop()
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        if send_work:
            send_work.wait()
            if self.log_pp:
                pipeline_log(bwd_mb_index, "bwd_send")

        # Return losses if there is a container passed in
        if self.log_pp:
            pipeline_log(0, "update", "start")
        self._stage.initialize()
        self._update_losses(self._stage, losses)

def _sorted_batch_p2p(
    p2p_ops: List[dist.P2POp], desc: Optional[str] = None
) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = dist.batch_isend_irecv(ops).pop()

    return work_by_peer