# Reward Shaping Notes for SoccerTwos

## Main issues
The current reward shaping provides useful dense signals, but it has several weaknesses. Ball progress is rewarded to both teammates equally, which makes credit assignment noisy. The proximity term is always negative, so agents are constantly punished unless they stay close to the ball, which may encourage blind ball chasing. The previous ball position is stored per player even though the ball state is global. In addition, the ball progress coefficient may be too large relative to the sparse scoring reward, and the existential penalty may accumulate too much over long episodes.

## Recommended changes
A better design is to keep the environment’s sparse score reward as the main objective and make all shaping terms small. The proximity term should reward improvement in distance to the ball rather than absolute distance. Ball progress should use one shared previous ball position and should mainly reward the closest player on each team, with at most a smaller shared reward for the teammate. If available, a small touch bonus can help agents discover ball interaction more quickly. A small “behind the ball” bonus can also encourage more sensible approach behavior. The existential penalty should be reduced significantly or removed at first.

## Practical suggestion
A simple and safer version is to replace absolute distance reward with distance delta, reduce the ball progress coefficient, use a single global previous ball x-position, and keep shaping much smaller than the true scoring reward. This should improve stability and reduce the chance that agents optimize dense shaping instead of actually learning to score and defend.

## Final takeaway
The main principle is that dense reward should guide exploration without overpowering the real task objective. In SoccerTwos, shaping based on small state changes usually works better than shaping based on absolute state values.