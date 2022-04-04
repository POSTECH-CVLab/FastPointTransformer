import os
import os.path as osp
import argparse

import numpy as np
from tqdm import tqdm

N = 41 # the number of rigid transforms
N_t = 26 # translation
N_r = 15 # rotation
M = 312 # the number of validation scenes


def main(args):
    ref_dir = osp.join(args.dir, "reference")
    ref_fnames = sorted(os.listdir(ref_dir))
    assert len(ref_fnames) == M
    pred_dirs = [osp.join(args.dir, f"transform{i}") for i in range(N)]
    
    print(">>> Loading references...")
    # load references    
    refs = [np.load(osp.join(ref_dir, fname)) for fname in ref_fnames]
    print("    done!")
    
    # translation
    print(">>> Calculating point-wise CScores for translation...")
    score_trans = [] # pointwise scores
    for i, (fname, ref) in enumerate(tqdm(zip(ref_fnames, refs))):
        score_tran = np.zeros_like(ref)
        for j in range(N_t):
            pred = np.load(osp.join(pred_dirs[j], fname))
            score_tran[np.where(ref == pred)] += 1
        score_tran = score_tran / N_t
        score_tran[np.where(ref == 255)] = -1
        score_trans.append(score_tran)
    print("    done!")

    # rotation
    print(">>> Calculating point-wise CScores for rotation...")
    score_rots = [] # pointwise scores
    for i, (fname, ref) in enumerate(tqdm(zip(ref_fnames, refs))):
        score_rot = np.zeros_like(ref)
        for j in range(N_r):
            pred = np.load(osp.join(pred_dirs[N_t + j], fname))
            score_rot[np.where(ref == pred)] += 1
        score_rot = score_rot / N_r
        score_rot[np.where(ref == 255)] = -1
        score_rots.append(score_rot)
    print("    done!")
        
    # full
    print(">>> Calculating point-wise CScores for full rigid transformations...")
    score_fulls = []
    for fname, score_tran, score_rot in tqdm(zip(ref_fnames, score_trans, score_rots)):
        score_full = (N_t*score_tran + N_r*score_rot) / N
        score_fulls.append(score_full)
    print("    done!")
        
    # final calculation
    cloudwise_score_trans = [np.mean(score_tran[np.where(score_tran > -0.5)]) for score_tran in score_trans]
    cloudwise_score_rots = [np.mean(score_rot[np.where(score_rot > -0.5)]) for score_rot in score_rots]
    cloudwise_score_fulls = [np.mean(score_full[np.where(score_full > -0.5)]) for score_full in score_fulls]
    
    if args.save:
        # save pointwise scores for visualization
        output_dir = osp.join(args.dir, "pointwise_scores")
        os.makedirs(output_dir, exist_ok=True)
        # save for visualization
        print(">>> Saving point-wise CScores for full rigid transformations...")
        for fname, cloudwise_score_full, score_full in tqdm(zip(ref_fnames, cloudwise_score_fulls, score_fulls)):
            scene_id = fname.split('.')[0]
            np.save(osp.join(output_dir, f'{scene_id}-score={round(1000 * cloudwise_score_full):d}.npy'), score_full)
        print("    done!")

    # final logging
    total_score_tran = np.mean(cloudwise_score_trans)
    total_score_rot = np.mean(cloudwise_score_rots)
    total_score_full = np.mean(cloudwise_score_fulls)
    print(">>> Results:")
    print(f"    Rotation: {100 * total_score_rot:.1f}")
    print(f"    Translation:: {100 * total_score_tran:.1f}")
    print(f"    Full: {100 * total_score_full:.1f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    main(args)