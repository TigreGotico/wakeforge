"""Extended tests for HMMStateExtractor."""
import pytest
import torch
import numpy as np
from ww_trainer.feats import MfccExtractor, HMMStateExtractor

def test_hmm_fit_updates_parameters():
    """Ensure that calling fit() actually modifies the HMM parameters."""
    pytest.importorskip("markovonnx")
    base = MfccExtractor(n_mfcc=13)
    ext = HMMStateExtractor(base, n_states=4, n_codes=8)
    ext.to("cpu")
    
    # Save initial (uniform) parameters
    pi_init = ext._pi.clone()
    A_init = ext._A.clone()
    B_init = ext._B.clone()
    
    # Create non-random patterns: repetitive sequences of the same sound
    # which should result in highly non-uniform HMM parameters
    pattern_audio = [torch.sin(torch.linspace(0, 100, 16000)) for _ in range(5)]
    
    ext.fit(pattern_audio, n_iter=5)
    
    # Verify parameters changed
    assert not torch.allclose(ext._pi, pi_init)
    assert not torch.allclose(ext._A, A_init)
    assert not torch.allclose(ext._B, B_init)
    assert ext._fitted is True

def test_hmm_forward_normalization():
    """Ensure that posteriors always sum to 1.0 even with random inputs."""
    base = MfccExtractor(n_mfcc=13)
    ext = HMMStateExtractor(base, n_states=8, n_codes=16)
    
    # Use random parameters (unfitted)
    out = ext(torch.randn(2, 16000))
    posteriors = out[..., 13:] # last 8 dims
    
    sums = posteriors.sum(dim=-1)
    np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-5)

def test_hmm_batch_consistency():
    """Ensure that batch processing gives the same result as individual processing."""
    base = MfccExtractor(n_mfcc=13)
    ext = HMMStateExtractor(base, n_states=4, n_codes=8)
    
    wavs = torch.randn(3, 16000)
    
    # Batch forward
    batch_out = ext(wavs)
    
    # Individual forwards
    indiv_outs = []
    for i in range(3):
        indiv_outs.append(ext(wavs[i:i+1]))
    indiv_out_cat = torch.cat(indiv_outs, dim=0)
    
    np.testing.assert_allclose(batch_out.detach().numpy(), indiv_out_cat.detach().numpy(), atol=1e-5)

def test_hmm_empty_sequence():
    """Ensure it handles short/empty sequences gracefully."""
    base = MfccExtractor(n_mfcc=13)
    ext = HMMStateExtractor(base, n_states=4, n_codes=8)
    
    # Very short audio (less than one hop)
    short_wav = torch.randn(1, 100)
    out = ext(short_wav)
    assert out.shape[1] >= 1
    assert out.shape[2] == 13 + 4

def test_hmm_device_transfer():
    """Ensure parameters move with the module."""
    base = MfccExtractor(n_mfcc=13)
    ext = HMMStateExtractor(base, n_states=4, n_codes=8)
    
    if torch.cuda.is_available():
        ext.cuda()
        assert ext._pi.device.type == "cuda"
        assert ext._A.device.type == "cuda"
        assert ext._codebook.device.type == "cuda"
        
        out = ext(torch.randn(1, 16000).cuda())
        assert out.device.type == "cuda"
    else:
        ext.cpu()
        assert ext._pi.device.type == "cpu"

def test_hmm_unfitted_behavior():
    """Ensure it works (with random features) even if not fitted."""
    base = MfccExtractor(n_mfcc=13)
    ext = HMMStateExtractor(base, n_states=4, n_codes=8)
    assert ext._fitted is False
    
    # Should not crash
    out = ext(torch.randn(1, 16000))
    assert out.shape[2] == 17
