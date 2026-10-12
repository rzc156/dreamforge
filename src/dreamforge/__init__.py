"""dreamforge: turn world-model imagination into corrective training data.

Pipeline (each stage is a CLI subcommand):

    mine     -> roll out a base VLA policy, record decision points + sim states
    imagine  -> optimize a corrective action chunk at each mined state with a
                latent world model (CEM over imagined futures)
    verify   -> replay mined states in the real simulator and measure whether
                imagined corrections actually change the outcome
                (arms: none / imagined / random)
    build    -> export (observation -> corrective chunk) samples, optionally
                filtered by verification, into a training set
    train    -> fine-tune the VLA on synthetic + replay data
    eval     -> paired closed-loop evaluation with confidence intervals
    report   -> aggregate everything into markdown + figures

The scientific question: test-time imagination (selecting among policy
samples) has near-zero headroom on a strong policy. Does *training-time*
imagination -- using a world model to synthesize corrective labels at the
policy's own failure states -- pay off where selection does not?
"""

__version__ = "0.1.0"
