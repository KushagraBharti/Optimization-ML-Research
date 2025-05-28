import math
from typing import List, Tuple, Dict, Optional

# === Environment Configuration ===
coverage_line_height = 2.0
battery_limit = 19.0
segments = [
    (-5.0, -3.0),
    (-2.0, -1.0),
    (1.0,  2.0),
    (4.0,  5.0),
    (7.0,  9.0),
]

# === Helper Functions ===

def tour_length(p: float, q: float, h: float) -> float:
    return math.hypot(p, h) + abs(q - p) + math.hypot(q, h)

def print_tour_details(p: float, q: float, h: float) -> float:
    O = (0.0, 0.0)
    P = (p, h)
    Q = (q, h)
    d1 = math.hypot(P[0] - O[0], P[1] - O[1])
    d2 = abs(Q[0] - P[0])
    d3 = math.hypot(Q[0] - O[0], Q[1] - O[1])
    total = d1 + d2 + d3
    print(f"      O → P = {O} → {P} = {d1:.4f}")
    print(f"      P → Q = {P} → {Q} = {d2:.4f}")
    print(f"      Q → O = {Q} → {O} = {d3:.4f}")
    print(f"        Total tour length = {total:.4f}")
    return total

# === Algorithm 3: DPOS (One-Sided DP) ===

def walkthrough_dpos(segments: List[Tuple[float, float]], h: float, L: float, side_label: str = "") -> Tuple[Optional[Dict[float, float]], Optional[Dict[float, str]]]:
    print(f"\n==========================")
    print(f"ALGORITHM 3: DPOS ({side_label})")
    print(f"==========================")
    print(f"Segments on line y = {h}, Battery limit L = {L}")
    print(f"Segments: {segments}")

    unreachable = [seg for seg in segments if tour_length(seg[0], seg[1], h) > L]
    if unreachable:
        print(f"\nUnreachable segments found: {unreachable}")
        return None, None

    C = sorted(set(b for a, b in segments))
    dp = {}
    back = {}

    a1 = segments[0][0]
    for c in C:
        print(f"\nEvaluating DP[{c:.2f}] — covering up to x = {c:.2f}")

        best_cost = float('inf')
        best_desc = ""

        # Case 1
        print(f"    Trying Case 1: One full tour from start of first segment ({a1:.2f}) to current end ({c:.2f})")
        len1 = tour_length(a1, c, h)
        print_tour_details(a1, c, h)
        if len1 <= L:
            best_cost = len1
            best_desc = f"Case 1: O → ({a1:.2f}, {h}) → ({c:.2f}, {h}) → O"

        # Case 2
        for i, (a_i, b_i) in enumerate(segments):
            if a_i <= c <= b_i:
                prev_end = segments[i - 1][1] if i > 0 else None
                prev_cost = dp.get(prev_end, 0.0)
                print(f"    Trying Case 2: Subinterval tour for segment ({a_i:.2f}, {b_i:.2f}) ending at {c:.2f}")
                len2 = tour_length(a_i, c, h)
                print_tour_details(a_i, c, h)
                if len2 <= L and prev_cost + len2 < best_cost:
                    best_cost = prev_cost + len2
                    best_desc = f"Case 2: DP[{prev_end}] + O → ({a_i:.2f}, {h}) → ({c:.2f}, {h}) → O"

        dp[c] = best_cost
        back[c] = best_desc
        print(f"DP[{c:.2f}] = {best_cost:.4f} via {best_desc}")

    print(f"\nFinal DP Value = {dp[C[-1]]:.4f} (minimum total length to cover all segments on {side_label})")
    return dp, back

# === Algorithm 4: General DP for Both Sides ===

def general_minlength_two_sided(segments: List[Tuple[float, float]], h: float, L: float):
    print("\n==========================")
    print("ALGORITHM 4: General DP (Both Sides)")
    print("==========================")
    print(f"All segments: {segments}")
    print(f"Line y = {h}, Battery limit L = {L}")

    left = [(-b, -a) for (a, b) in segments if b < 0]
    right = [(a, b) for (a, b) in segments if a > 0]

    print(f"\nLeft-side segments (reflected to +x): {left}")
    print(f"Right-side segments: {right}")

    dp_left, _ = walkthrough_dpos(left, h, L, side_label="LEFT SIDE")
    dp_right, _ = walkthrough_dpos(right, h, L, side_label="RIGHT SIDE")

    if dp_left is None or dp_right is None:
        print("\nAborting: One side has unreachable segments.\n")
        return

    max_l = max(dp_left)
    max_r = max(dp_right)
    total_cost = dp_left[max_l] + dp_right[max_r]
    print(f"\nTotal cost (no bridge): {dp_left[max_l]:.4f} + {dp_right[max_r]:.4f} = {total_cost:.4f}")

    best = total_cost
    C_l = sorted(dp_left.keys())
    C_r = sorted(dp_right.keys())

    print("\nEvaluating possible bridge tours between left and right sides:")
    for p in C_l:
        for q in C_r:
            length_pq = tour_length(p, q, h)
            if length_pq <= L:
                combined = dp_left[p] + length_pq + dp_right[q]
                print(f"  Bridge tour: left={-p:.2f}, right={q:.2f} → length={length_pq:.4f}, total={combined:.4f}")
                if combined < best:
                    best = combined

    print(f"\nFinal Optimal Cost (with or without bridge) = {best:.4f}\n")

# === Main Execution ===
if __name__ == "__main__":
    right_side_segments = [(a, b) for (a, b) in segments if a > 0]
    walkthrough_dpos(right_side_segments, coverage_line_height, battery_limit, side_label="RIGHT SIDE")
    general_minlength_two_sided(segments, coverage_line_height, battery_limit)
