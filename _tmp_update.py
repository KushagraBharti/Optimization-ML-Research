from pathlib import Path
path = Path('dp_one_side_ref.py')
text = path.read_text()
text = text.replace('__all__ = ["generate_candidates_one_side", "dp_one_side"]', '__all__ = ["generate_candidates_one_side", "dp_one_side", "generate_candidates_one_side_ref", "dp_one_side_ref"]', 1)
insertion = '\n\n# Public reference-entry wrappers\ndef generate_candidates_one_side_ref(segments: List[Tuple[float, float]], h: float, L: float) -> List[float]:\n    return generate_candidates_one_side(segments, h, L)\n\n\ndef dp_one_side_ref(segments: List[Tuple[float, float]], h: float, L: float) -> Tuple[List[float], List[float]]:\n    return dp_one_side(segments, h, L)\n'
marker = '\n\nif __name__ == "__main__":\n'
if marker not in text:
    raise SystemExit('marker not found')
text = text.replace(marker, insertion + marker, 1)
path.write_text(text)
