from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, logprobs, *args, **kwargs):

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    local_logprob = ys[q,c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
        done_beams = []

        for t in range(self.seq_length):

            logprobsf = logprobs.data.float() # lets go to CPU for more efficiency in indexing operations
            logprobsf[:,logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000  
        
            beam_seq,\
            beam_seq_logprobs,\
            beam_logprobs_sum,\
            state,\
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(), 
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    beam_logprobs_sum[vix] = -1000

            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams