import math

# === Environment Configuration ===
coverage_line_height = 2.0  # y = h
battery_limit = 18.0        # L
segments = [                # List of segments for GS: (a, b)
    (1.0, 2.0),
    (4.0, 5.0),
    (7.0, 8.0),
    (9.0, 10.0),
]
single_segment = (2.0, 6.0)  # Segment for GSP: (a, b)

# === Helper Functions ===

def distance(p1, p2):
    """Compute and print Euclidean distance between two points."""
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    print(f"    Distance {p1} -> {p2} = {dist:.4f}")
    return dist

def horizontal_distance(x1, x2):
    """Compute and print horizontal distance on coverage line."""
    dist = abs(x2 - x1)
    print(f"    Horizontal distance x={x1:.4f} -> x={x2:.4f} = {dist:.4f}")
    return dist

def tour_length_detailed(p, q, h):
    """Print breakdown and return length of tour O->P->Q->O."""
    O = (0.0, 0.0)
    P = (p, h)
    Q = (q, h)
    print("  Leg 1: O -> P")
    d1 = distance(O, P)
    print("  Leg 2: P -> Q (horizontal)")
    d2 = horizontal_distance(p, q)
    print("  Leg 3: Q -> O")
    d3 = distance(Q, O)
    total = d1 + d2 + d3
    print(f"  Total length = {d1:.4f} + {d2:.4f} + {d3:.4f} = {total:.4f}\n")
    return total

def find_maximal_p(q, h, L):
    """
    Solve for p such that O->(p,h)->(q,h)->O == L.
    Raises ValueError if no solution exists.
    """
    r = math.hypot(q, h)
    # shortest closed tour is P=Q: length = 2*r
    if 2*r > L:
        raise ValueError(f"Endpoint (q={q},h={h}) unreachable: need >= {2*r:.4f}, have {L:.4f}")
    K = L - r - q
    p = (h*h - K*K) / (2*K)
    return p

def subtract_covered_intervals(segments, covered):
    """Subtract covered interval (p,q) from all segments list."""
    p, q = covered
    updated = []
    for a, b in segments:
        if b <= p or a >= q:
            updated.append((a, b))
        else:
            if a < p < b:
                updated.append((a, p))
            if a < q < b:
                updated.append((q, b))
    return updated

# === GS: Minimize Number of Tours ===

def greedy_min_tours(segments, h, L):
    print("=== Greedy Strategy (GS): Minimize Number of Tours ===")
    print(f"Line y={h}, Battery L={L:.4f}\nSegments: {segments}\n")
    
    # Pre-filter unreachable segments
    unreachable = [seg for seg in segments if 2*math.hypot(seg[1], h) > L]
    if unreachable:
        print(f"WARNING: Unreachable segments (skipping): {unreachable}")
        segments = [seg for seg in segments if seg not in unreachable]
        print(f"Remaining segments: {segments}\n")
    
    remaining = sorted(segments, key=lambda s: s[1])
    tour_count = 0
    
    while remaining:
        tour_count += 1
        print(f"--- Tour {tour_count} ---")
        # pick farthest reachable endpoint
        farthest = remaining[-1][1]
        leftmost = remaining[0][0]
        print(f"Farthest endpoint x={farthest:.4f}")
        
        # try minimal-length covering all
        print("Attempt minimal-length tour:")
        minimal_len = tour_length_detailed(leftmost, farthest, h)
        
        if minimal_len <= L:
            p, q = leftmost, farthest
            print("Using minimal-length tour to finish.")
        else:
            print("Minimal-length too long, doing maximal-length tour:")
            q = farthest
            p = find_maximal_p(q, h, L)
            print(f"Computed P=({p:.4f},{h:.4f})")
            _ = tour_length_detailed(p, q, h)
        
        covered = (min(p, q), max(p, q))
        print(f"Tour covers interval {covered}")
        remaining = subtract_covered_intervals(remaining, covered)
        print(f"Remaining: {remaining}\n")
    
    print(f"Total tours: {tour_count}\n")

# === GSP: Single Segment MinLength ===

def greedy_min_length_one_segment(seg, h, L):
    print("=== GSP: MinLength (1 segment) ===")
    print(f"Line y={h}, Projection O'=(0,{h})\nSegment: {seg}, L={L:.4f}\n")
    a, b = seg
    
    # check reachability
    if 2*math.hypot(b, h) > L:
        print(f"ERROR: Segment endpoint b={b} unreachable on one tour.")
        return
    
    # step 1: minimal-length whole
    print("Step 1: minimal-length for [a,b]:")
    full_len = tour_length_detailed(a, b, h)
    if full_len <= L:
        print("Whole segment done in 1 tour.\n")
        return
    
    # step 2: maximal from b
    print("Step 2: maximal-length from Q=b:")
    p1 = find_maximal_p(b, h, L)
    print(f"P1=({p1:.4f},{h:.4f})")
    _ = tour_length_detailed(p1, b, h)
    
    # remaining
    rem = (a, p1)
    print(f"Remaining interval: {rem}\n")
    
    # step 3: leftover
    print("Step 3: minimal-length for leftover:")
    rem_len = tour_length_detailed(rem[0], rem[1], h)
    if rem_len <= L:
        print("Leftover done in 1 more tour.\n")
    else:
        print("Leftover too large, splitting at O':")
        print("Tour 2: left to projection")
        _ = tour_length_detailed(rem[0], 0, h)
        print("Tour 3: projection to right")
        _ = tour_length_detailed(0, rem[1], h)
        print("Completed in 3 tours.\n")

# === Main ===

if __name__ == "__main__":
    greedy_min_tours(segments, coverage_line_height, battery_limit)
    greedy_min_length_one_segment(single_segment, coverage_line_height, battery_limit)

