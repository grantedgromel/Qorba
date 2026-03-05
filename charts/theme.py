"""Plotly layout defaults for the Qorba neobrutalist theme.

Provides a consistent visual language across all charts:
- Inter font family
- White plot background, light gray paper background
- No gridlines
- Navy axis labels
- Stripe-inspired color palette
"""

from config.settings import COLORS, CHART_COLORS


def get_color_sequence() -> list[str]:
    """Return the standard chart color sequence."""
    return list(CHART_COLORS)


def get_layout(**overrides) -> dict:
    """Return a dict of Plotly layout defaults for fig.update_layout().

    Pass keyword arguments to override any default. For example:
        fig.update_layout(**get_layout(title="My Chart", height=500))
    """
    defaults = {
        # ── Typography ────────────────────────────────────────────────
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "size": 13,
            "color": COLORS["navy_900"],
        },
        "title": {
            "font": {
                "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "size": 18,
                "weight": 800,
                "color": COLORS["navy_900"],
            },
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"l": 10, "t": 10},
        },

        # ── Background ───────────────────────────────────────────────
        "plot_bgcolor": COLORS["white"],
        "paper_bgcolor": COLORS["gray_100"],

        # ── X-Axis ───────────────────────────────────────────────────
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linewidth": 1.5,
            "linecolor": COLORS["gray_200"],
            "tickfont": {
                "family": "Inter, sans-serif",
                "size": 11,
                "color": COLORS["navy_700"],
            },
            "title_font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": COLORS["navy_700"],
                "weight": 600,
            },
        },

        # ── Y-Axis ───────────────────────────────────────────────────
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linewidth": 1.5,
            "linecolor": COLORS["gray_200"],
            "tickfont": {
                "family": "Inter, sans-serif",
                "size": 11,
                "color": COLORS["navy_700"],
            },
            "title_font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": COLORS["navy_700"],
                "weight": 600,
            },
        },

        # ── Legend ────────────────────────────────────────────────────
        "legend": {
            "font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": COLORS["navy_900"],
            },
            "bgcolor": "rgba(255,255,255,0)",
            "borderwidth": 0,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },

        # ── Margins ──────────────────────────────────────────────────
        "margin": {"l": 60, "r": 30, "t": 70, "b": 50},

        # ── Hover ────────────────────────────────────────────────────
        "hoverlabel": {
            "bgcolor": COLORS["navy_900"],
            "font_size": 12,
            "font_family": "Inter, sans-serif",
            "font_color": COLORS["white"],
            "bordercolor": COLORS["navy_900"],
        },
        "hovermode": "x unified",

        # ── Color ────────────────────────────────────────────────────
        "colorway": CHART_COLORS,
    }

    # Apply overrides (shallow merge at top level)
    defaults.update(overrides)
    return defaults
