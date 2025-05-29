# examples/run_all.py
from coverage_planning.greedy   import greedy_min_tours
from coverage_planning.gsp      import greedy_min_length_one_segment
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.dp_both  import dp_full_line

def main():
    h = 2.0

    # 1) Greedy MinTours
    print("\n=================================================================================================================================================")
    print("Algorithm 1: Greedy MinTours")
    print("=================================================================================================================================================\n")
    segs1 = [
        (1.0, 2.0),
        (4.0, 5.0),
        (7.0, 8.0),
    ]
    L1   = 17.0
    cnt, tours = greedy_min_tours(segs1, h, L1)
    print("--> Greedy MinTours →", cnt, tours)

    # 2) GSP single-segment MinLength
    print("\n=================================================================================================================================================")
    print("Algorithm 2: GSP MinTours")
    print("=================================================================================================================================================\n")
    seg2 = (2,6); L2 = 13.0
    cnt2, tours2 = greedy_min_length_one_segment(seg2, h, L2)
    print("--> GSP Single-Segment →", cnt2, tours2)

    # 3) One-sided DPOS
    print("\n=================================================================================================================================================")
    print("Algorithm 3: DP MinLength One-Sided")
    print("=================================================================================================================================================\n")
    segs3 = [(1,2),(4,5),(7,9)]; L3 = 19.0
    dp_vals, back = dp_one_side(segs3, h, L3)
    print("--> One-Side DPOS → cost:", dp_vals[-1], "states:", len(dp_vals))

    # 4) Two-sided DP
    print("\n=================================================================================================================================================")
    print("Algorithm 4: DP MinLength Both-Sides")
    print("=================================================================================================================================================\n")
    segs4 = [(-5,-3),(-2,-1),(1,2),(4,5),(7,9)]; L4 = 19.0
    opt = dp_full_line(segs4, h, L4)
    print("--> Two-Side MinLength DP → cost:", opt, "\n")

if __name__=="__main__":
    main()
