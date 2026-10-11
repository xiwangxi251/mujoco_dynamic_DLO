# Fixed first-three-second oracle for ordinary PPO

This is an exploratory comparison against the existing 151 ordinary PPO, with
unchanged normal actuator limits, 5-D free actions and session_height_v1 reward.
The 151 training used different promotion mechanisms before approximately 3M;
the new scratch20M group uses fast promotion throughout. It is not a pure
single-variable causal comparison. Additional input increases first-layer
parameter count; hidden architecture remains actor/critic 256/256/128 ReLU.

`--future-geometry` requires ordinary state observations. The original state99
is preserved. The added 30 frames are at episode time .1,.2,...,3 seconds.
Each frame includes 14 corresponding material point xyz coordinates, relative
time / 3 and an availability mask: 99 + 30*44 = 1419 dimensions.

A separate copied full model/state/drive/RNG branch precomputes once per reset.
Robot contacts are disabled in the branch, the robot is held at reset pose,
while cable/table dynamics and active replay constraints remain unchanged.
This is a conditional no-robot-contact natural-motion reference, not the
realized future after the policy makes contact. Hidden drive information is
available to the oracle. No branch physics samples count toward PPO's budget.

Future material interpolation is fixed from the initial arc-length samples;
each frame does not select fresh material nodes. World trajectories remain
fixed within a round; each observation transforms into the current TCP frame.
Expired absolute slots are zeroed. First robot/cable contact at any physics
substep permanently masks the remaining future for this episode; release or
retry does not reactivate it. After 3s the future is exhausted while the
original 15s task continues. Future masks do not use actual future contact time.

The wrapper is present in both train and evaluation factories. No separate
encoder, goal selector, planned-contact-time action, future reward or change
to the underlying success criterion is introduced. The ordinary current-only
control group remains unmodified.

Validation includes fixed material correspondence, coordinate/time/mask
semantics, seven scene reset/step live physics and RNG isolation, contact latch,
and short precontact forecast position accuracy. Server deployment additionally
requires spawned worker/PPO smoke, saved model reload, actual failure exit
classification, and post-launcher fresh worker-ready/configuration/identity
confirmation. Training is independently supervised with heartbeat and real
progress deadlines and exact cumulative 20M samples.
