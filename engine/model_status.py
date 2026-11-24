"""User-facing scientific-status summaries without application dependencies."""

from engine.forward import resolve_strike_mode


def model_status_lines(bodies):
    """Return precise method labels for gravity and magnetic outputs."""
    modes = sorted({resolve_strike_mode(body.y_min, body.y_max,
                                       getattr(body, "strike_mode", None))
                    for body in bodies})
    lines = []
    for mode in modes:
        if mode == "infinite":
            lines.append(
                "Infinite Strike: gravity uses the benchmarked 2D positive-down kernel; "
                "magnetics use the benchmarked 2D boundary-charge kernel."
            )
        else:
            lines.append(
                "Finite Strike: gravity and magnetics use closed-form polygonal-prism "
                "face integrals, validated against rectangular-prism and independent "
                "volume-integration references."
            )
    return lines or ["No bodies are present in this project."]
