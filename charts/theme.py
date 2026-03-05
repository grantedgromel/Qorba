"""Plotly layout defaults for the Qorba dark theme.

Provides a consistent visual language across all charts:
- Inter font family
- Transparent plot/paper background (blends into dark page)
- Subtle axis lines
- Light gray labels on dark background
- Dark-mode-friendly color palette
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
        # -- Typography ────────────────────────────────────────────────
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "size": 13,
            "color": COLORS["text_secondary"],
        },
        "title": {
            "font": {
                "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "size": 18,
                "weight": 700,
                "color": COLORS["text_primary"],
            },
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"l": 10, "t": 10},
        },

        # -- Background (transparent to blend into dark page) ─────────
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",

        # -- X-Axis ───────────────────────────────────────────────────
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linewidth": 1,
            "linecolor": COLORS["border"],
            "tickfont": {
                "family": "Inter, sans-serif",
                "size": 11,
                "color": COLORS["text_secondary"],
            },
            "title_font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": COLORS["text_secondary"],
                "weight": 600,
            },
        },

        # -- Y-Axis ───────────────────────────────────────────────────
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linewidth": 1,
            "linecolor": COLORS["border"],
            "tickfont": {
                "family": "Inter, sans-serif",
                "size": 11,
                "color": COLORS["text_secondary"],
            },
            "title_font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": COLORS["text_secondary"],
                "weight": 600,
            },
        },

        # -- Legend ────────────────────────────────────────────────────
        "legend": {
            "font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": COLORS["text_secondary"],
            },
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },

        # -- Margins ──────────────────────────────────────────────────
        "margin": {"l": 60, "r": 30, "t": 70, "b": 50},

        # -- Hover ────────────────────────────────────────────────────
        "hoverlabel": {
            "bgcolor": COLORS["bg_elevated"],
            "font_size": 12,
            "font_family": "Inter, sans-serif",
            "font_color": COLORS["text_primary"],
            "bordercolor": COLORS["border"],
        },
        "hovermode": "x unified",

        # -- Color ────────────────────────────────────────────────────
        "colorway": CHART_COLORS,
    }

    # Apply overrides (shallow merge at top level)
    defaults.update(overrides)
    return defaults
