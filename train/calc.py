import math
import re
import math
from typing import Union
from collections import namedtuple

class IsacCalculator:
    """
    IsacCalculator provides various scaled per-hour computations based on gameplay stats such as hits,
    headshots, critical hits, and kills. It also includes utility functions for time normalization
    and converting string-based time units to total hours.
    """

    Interval = namedtuple("Interval", ["min", "max", "multiplier"])
    DIVISOR_INTERVALS = [
        Interval(0, 100, 3.0),
        Interval(100, 200, 2.5),
        Interval(200, 300, 2.0),
        Interval(300, 400, 1.5),
        Interval(400, 600, 1.2),
        Interval(600, 800, 1.1),
    ]

    def __init__(
        self,
        time_played_total: int = 0,
        headshots: int = 0,
        sum_hits: int = 0,
        sum_critical_hits: int = 0,
        kills_npc: int = 0,
        kills_headshot: int = 0
    ):
        """Initialize the calculator with gameplay statistics."""
        self.time_played_total = time_played_total
        self.headshots = headshots
        self.sum_hits = sum_hits
        self.sum_critical_hits = sum_critical_hits
        self.kills_npc = kills_npc
        self.kills_headshot = kills_headshot
        self.bodyshots = max(self.sum_hits - self.headshots, 0)

    def _get_divisor(self) -> float:
        """Determine scaling divisor based on total play time and pre-defined intervals."""
        for interval in self.DIVISOR_INTERVALS:
            if interval.min <= self.time_played_total < interval.max:
                return self.time_played_total * interval.multiplier
        return self.time_played_total

    def _safe_divide(self, numerator: int) -> int:
        """Safely divide the given value by the calculated divisor."""
        divisor = self._get_divisor()
        return math.floor(numerator / divisor) if divisor > 0 else 0

    def calculate_scaled_hits_per_hour(self) -> int:
        """Calculate weapon hits per hour, scaled by play time."""
        return self._safe_divide(self.sum_hits)

    def calculate_scaled_critical_hits_per_hour(self) -> int:
        """Calculate critical hits per hour, scaled by play time."""
        return self._safe_divide(self.sum_critical_hits)

    def calculate_scaled_headshots_per_hour(self) -> int:
        """Calculate headshots per hour, scaled by play time."""
        return self._safe_divide(self.headshots)

    def calculate_scaled_bodyshots_per_hour(self) -> int:
        """Calculate bodyshots per hour, scaled by play time."""
        return self._safe_divide(self.bodyshots)

    def calculate_scaled_npc_kills_per_hour(self) -> int:
        """Calculate NPC kills per hour, scaled by play time."""
        return self._safe_divide(self.kills_npc)

    def calculate_scaled_headshot_kills_per_hour(self) -> int:
        """Calculate headshot kills per hour, scaled by play time."""
        return self._safe_divide(self.kills_headshot)

    def time_unit_convert_to_hours(self, input_value: Union[str, None]) -> int:
        """
        Convert a time string (e.g. "2h30m") to total hours as an integer.
        Accepts strings with hours (h), minutes (m), and seconds (s).
        """
        if not input_value:
            return 0

        if isinstance(input_value, int):
            return input_value

        input_value = re.sub(r"[^a-zA-Z0-9]", "", input_value)

        hours = int(re.search(r"(\d+)h", input_value).group(1)) if re.search(r"(\d+)h", input_value) else 0
        minutes = int(re.search(r"(\d+)m", input_value).group(1)) if re.search(r"(\d+)m", input_value) else 0
        seconds = int(re.search(r"(\d+)s", input_value).group(1)) if re.search(r"(\d+)s", input_value) else 0

        total_hours = hours + (minutes / 60) + (seconds / 3600)
        return round(total_hours)

    def calculate_bodyshots_per_hour(self) -> int:
        """Return bodyshots per hour if valid for scaling."""
        return self.calculate_scaled_bodyshots_per_hour() if self.bodyshots > self.time_played_total else 0

    def calculate_headshots_per_hour(self) -> int:
        """Return headshots per hour if valid for scaling."""
        return self.calculate_scaled_headshots_per_hour() if self.time_played_total > 0 and self.headshots > 0 else 0

    def calculate_npc_kills_per_hour(self) -> int:
        """Return NPC kills per hour if valid for scaling."""
        return self.calculate_scaled_npc_kills_per_hour() if self.time_played_total > 0 and self.kills_npc > 0 else 0

    def calculate_headshot_kills_per_hour(self) -> int:
        """Return headshot kills per hour if valid for scaling."""
        return self.calculate_scaled_headshot_kills_per_hour() if self.time_played_total > 0 and self.kills_headshot > 0 else 0

    def calculate_weapon_hits_per_hour(self) -> int:
        """Return weapon hits per hour if valid for scaling."""
        if self.time_played_total > 0 and self.sum_hits > self.time_played_total:
            return self.calculate_scaled_hits_per_hour()
        return 0

    def calculate_critical_hits_per_hour(self) -> int:
        """Return critical hits per hour if valid for scaling."""
        if self.time_played_total > 0 and self.sum_critical_hits > self.time_played_total:
            return self.calculate_scaled_critical_hits_per_hour()
        return 0

    def calculate_total_headshot_kills(self, total_kills=0, total_headshots=0, avg_bullets_to_kill=50, avg_crit_chance=30) -> int:
        """
        Estimate total headshot kills based on average bullets to kill and critical hit chance.

        Formula:
            H = (T / A) * P * (headshots / T) * A

            Where:
                H = Estimated number of headshot kills
                B = Number of bodyshot kills
                C = Number of critical hits
                T = Total kills
                A = Average bullets to kill
                P = Crit chance (as a decimal)

        Parameters:
            total_kills (int): Total kills made.
            total_headshots (int): Number of headshots.
            avg_bullets_to_kill (int): Average bullets needed to kill.
            avg_crit_chance (int): Average critical hit chance in percent.

        Returns:
            int: Estimated headshot kills.
            Total kills made.
            total_headshots (int): Number of headshots.
            avg_bullets_to_kill (int): Average bullets needed to kill.
            avg_crit_chance (int): Average critical hit chance in percent.

        Returns:
            int: Estimated headshot kills.
        """
        if total_kills and total_headshots:
            return math.ceil(
                (total_kills / avg_bullets_to_kill)
                * (avg_crit_chance / 100)
                * (total_headshots / total_kills)
                * avg_bullets_to_kill
            )
        return 0

    def calculate_percentages_and_ratios(self):
        try:
            headshots = int(self.headshots)
            sum_hits = int(self.sum_hits)
            time_played = int(self.time_played_total)

            if headshots <= 0 or sum_hits <= 0 or time_played <= 0:
                return None

            bodyshots = sum_hits - headshots
            if bodyshots < 0:
                bodyshots = 0

            hs_percentage = (headshots / sum_hits) * 100
            bs_percentage = (bodyshots / sum_hits) * 100

            hs_to_bs_ratio = headshots / (bodyshots + 1)
            bs_to_hs_ratio = bodyshots / (headshots + 1)

            return {
                "percentageOfHeadshots": round(hs_percentage, 2),
                "percentageOfBodyshots": round(bs_percentage, 2),
                "totalHeadshots": headshots,
                "totalBodyshots": bodyshots,
                "hsToBsRatio": round(hs_to_bs_ratio, 2),
                "bsToHsRatio": round(bs_to_hs_ratio, 2),
            }

        except (ValueError, ZeroDivisionError):
            return None
