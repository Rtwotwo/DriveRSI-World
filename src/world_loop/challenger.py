"""Small-loop adversarial curriculum primitives, not a claim of learned adversaries.

Candidates need matched current/reference rollouts and separate geometry/render
qualification before their score can enter a training queue. Failed references
are rejected instead of rewarding unavoidable collisions.
"""
import math
from driversi_world.simulation.closed_loop.challenges import digest


def candidate_requests(capability, seed, limit=12):
    """Deterministic bounded search; selection from measured regret is a separate step."""
    if type(seed) is not int or not 0 <= seed < 2**32 or type(limit) is not int or not 1 <= limit <= 48:
        raise ValueError('Bounded seed and candidate budget required')
    result = []
    for actor,description in sorted(capability['actors'].items()):
        if description['initial_overlap_with']:
            continue
        for scale in (.5,1.5):
            for start in (3,6):
                result.append(dict(schema='driversi.world-challenge.v1',request_id=f'challenge-{seed}-{len(result):03d}',
                    scene_id=capability['scene_id'],scene_sha256=capability['scene_sha256'],
                    capability_sha256=capability['capability_sha256'],split='train',seed=seed,
                    objective='reference_regret',edit=dict(actor_id=actor,desired_speed_scale=scale,
                                                         start_step=start,duration_steps=3)))
                if len(result) == limit:
                    return result
    return result


def qualify_pair(request, current, reference, validation):
    """Score only complete matched rollout evidence from a trusted evaluator.

    This pure validator is not an artifact authenticator: the calling evaluator
    must verify trace hashes. Retain raw safety metrics separately from the score.
    """
    required = ('request_sha256','initial_state_sha256','execution_contract_sha256',
                'policy_sha256','seed','executed_steps','complete','collision_count',
                'max_offroad_fraction','progress_m')
    for item in (current,reference):
        if any(k not in item for k in required) or item['complete'] is not True:
            raise ValueError('Complete current/reference rollout evidence required')
        for key in ('collision_count','max_offroad_fraction','progress_m'):
            if type(item[key]) not in (int,float) or not math.isfinite(item[key]) or item[key] < 0:
                raise ValueError('Invalid raw rollout metric')
        if not 0 <= item['max_offroad_fraction'] <= 1 or type(item['collision_count']) is not int:
            raise ValueError('Invalid safety metric range')
        if item['request_sha256'] != digest(request) or item['seed'] != request['seed']:
            raise ValueError('Rollout does not belong to this challenge')
        if type(item['executed_steps']) is not int or not 1 <= item['executed_steps'] <= 8:
            raise ValueError('Executed steps must be within the eight-step horizon')
        if (item['executed_steps'] < 8 and item['collision_count'] == 0
                and item['max_offroad_fraction'] <= .2):
            raise ValueError('Short rollout without environment termination is incomplete')
    for key in ('initial_state_sha256','execution_contract_sha256','seed'):
        if current[key] != reference[key]:
            raise ValueError('Unmatched current/reference conditions')
    if current['policy_sha256'] == reference['policy_sha256']:
        raise ValueError('A distinct fixed reference policy is required')
    # A terminal failure can shorten current rollout; the successful reference
    # must cover the full eight post-warmup intervals before claiming solvability.
    gates = ('trace_hashes_verified','training_split_verified','geometry_valid','rendering_valid')
    missing = [k for k in gates if validation.get(k) is not True]
    reference_safe = (reference['collision_count'] == 0 and reference['max_offroad_fraction'] <= .01
                      and reference['progress_m'] >= 1. and reference['executed_steps'] == 8)
    if missing or not reference_safe:
        return dict(eligible=False,score=None,reason='unqualified_reference_or_evidence',
                    missing=missing,reference_safe=reference_safe,training_ready=False)
    def cost(row):
        # Progress is bounded so travelling farther cannot erase a collision.
        progress_credit = min(row['progress_m']/reference['progress_m'],1.)
        return 10.*min(row['collision_count'],1) + 5.*row['max_offroad_fraction'] - progress_credit
    regret = max(0.,cost(current)-cost(reference))
    return dict(eligible=True,score=regret,reference_safe=True,
        training_ready=True,reason='qualified_empirical_regret',
        scope='one paired rollout; empirical reference success is not a proof of solvability')
