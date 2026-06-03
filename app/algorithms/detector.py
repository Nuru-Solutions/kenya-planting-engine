"""
app/algorithms/detector.py

Signal 1 – Rainfall Onset  (40%): CHIRPS daily, Modified Stern Algorithm
Signal 2 – NDVI Greenup    (35%): Sentinel-2, Savitzky-Golay smoothed
Signal 3 – SAR Tillage     (25%): Sentinel-1 VV/VH backscatter change

The NDVI and SAR signals also extract full phenological profiles
(peak NDVI, season length, NDVI integral, SAR at peak, raw time series)
for direct consumption by the downstream crop identification pipeline.
"""
from __future__ import annotations
import logging
import statistics
from datetime import date, timedelta
from typing import Optional

from app.core.config import (
    get_settings, AEZConfig, Season, SeasonWindow, ConfidenceLevel,
)
from app.core.models import (
    NDVIObservation, SARObservation, RainfallRecord,
    RainfallOnsetSignal, NDVIGreenupSignal, SARTillageSignal,
)

logger = logging.getLogger(__name__)


# ── Signal 1: Rainfall Onset ───────────────────────────────────────────────────

class RainfallOnsetDetector:
    """
    Modified Stern Algorithm:
    1. Find first 3 consecutive days with cumulative rain >= threshold (25mm)
    2. False-start check: >7 dry days within 14 days after onset
    3. Confidence weighted by rain amount, false-start penalty, window fit
    """

    def __init__(self):
        self.settings = get_settings()

    def detect(
        self,
        records: list[RainfallRecord],
        season_window: SeasonWindow,
        year: int,
    ) -> RainfallOnsetSignal:
        if not records:
            return RainfallOnsetSignal(available=False)

        thr        = self.settings.rainfall_onset_threshold_mm
        fs_window  = self.settings.rainfall_false_start_days
        win_start, win_end = season_window.get_window(year)
        search_start       = season_window.get_search_start(year)

        rmap  = {r.record_date: r.rainfall_mm for r in records}
        dates = sorted(d for d in rmap if search_start <= d <= win_end)

        if len(dates) < 3:
            return RainfallOnsetSignal(available=False)

        seasonal_total = round(sum(rmap.get(d, 0.0) for d in dates), 1)

        candidates = []
        for i in range(len(dates) - 2):
            d0, d1, d2 = dates[i], dates[i+1], dates[i+2]
            if (d1 - d0).days != 1 or (d2 - d1).days != 1:
                continue
            cum = rmap[d0] + rmap[d1] + rmap[d2]
            if cum >= thr:
                dry = self._dry_days_after(rmap, dates, d2, fs_window)
                fs  = dry > 7
                conf = self._score(cum, dry, d0, win_start, win_end, thr)
                candidates.append({
                    "onset_date": d0, "cumulative_3day_mm": round(cum, 2),
                    "dry_days_after": dry, "is_false_start": fs, "confidence": conf,
                })

        if not candidates:
            return RainfallOnsetSignal(
                available=True, confidence=0.1, is_false_start=True,
                total_seasonal_rainfall_mm=seasonal_total,
            )

        real = [c for c in candidates if not c["is_false_start"]]
        best = (real or candidates)[0]

        return RainfallOnsetSignal(
            onset_date=best["onset_date"],
            cumulative_3day_mm=best["cumulative_3day_mm"],
            is_false_start=best["is_false_start"],
            dry_spell_within_14d=best["dry_days_after"],
            total_seasonal_rainfall_mm=seasonal_total,
            confidence=best["confidence"],
            available=True,
        )

    def _dry_days_after(self, rmap, dates, after, window, dry_thr=1.0):
        end = after + timedelta(days=window)
        return sum(1 for d in dates if after < d <= end and rmap.get(d, 0.0) < dry_thr)

    def _score(self, cum, dry, onset, win_start, win_end, thr):
        rain_s = min(1.0, (cum - thr) / (thr * 2) + 0.5)
        # 0.07/day penalty (was 0.10) — Kenya rainfall is episodic; 4 dry days
        # within 14d of onset is normal and shouldn't cut confidence by 40%
        dry_p  = max(0.0, 1.0 - dry * 0.07)
        if win_start <= onset <= win_end:
            win_b = 1.0
        else:
            out   = max(0, (win_start - onset).days, (onset - win_end).days)
            win_b = max(0.4, 1.0 - out / 30)
        return round(rain_s * dry_p * win_b, 3)


# ── Signal 2: NDVI Greenup + Full Phenological Profile ────────────────────────

class NDVIGreenupDetector:
    """
    Planting date detection:
      - Filter obs by cloud < 60%, pixel_count >= 5
      - Pre-season baseline NDVI (8 weeks before window)
      - Savitzky-Golay smoothing (falls back to moving average)
      - First confirmed NDVI rise > 0.08 above baseline
      - Planting = greenup − 12 days (maize emergence offset)

    Phenological profile (crop ID pipeline):
      peak_ndvi, peak_date, senescence_date, season_length_days,
      ndvi_at_harvest, ndvi_integral, ndvi_rise_rate, ndvi_timeseries
    """

    PLANTING_OFFSET = 12   # days from greenup to planting (maize)
    MIN_PIXELS      = 3    # was 5 — small farm polygons at 10m need a lower floor
    MAX_CLOUD       = 60

    def __init__(self):
        self.settings = get_settings()

    def detect(
        self,
        observations: list[NDVIObservation],
        season_window: SeasonWindow,
        year: int,
    ) -> NDVIGreenupSignal:
        if not observations:
            return NDVIGreenupSignal(available=False)

        win_start, win_end = season_window.get_window(year)
        search_start       = season_window.get_search_start(year)

        valid = [o for o in observations
                 if o.cloud_cover_pct <= self.MAX_CLOUD and o.pixel_count >= self.MIN_PIXELS]

        if len(valid) < 3:
            return NDVIGreenupSignal(
                available=True, confidence=0.1,
                cloud_gap_days=self._max_gap(observations, search_start, win_end),
                n_observations=len(valid),
            )

        # Pre-season baseline
        pre_obs  = [o for o in valid
                    if (win_start - timedelta(days=56)) <= o.obs_date < search_start]
        baseline = statistics.median([o.ndvi for o in (pre_obs or valid[:3])])

        # In-window observations
        in_win = sorted([o for o in valid if search_start <= o.obs_date <= win_end],
                        key=lambda x: x.obs_date)

        if len(in_win) < 2:
            return NDVIGreenupSignal(
                available=True, baseline_ndvi=round(baseline, 4),
                confidence=0.15, n_observations=len(in_win),
            )

        dates   = [o.obs_date for o in in_win]
        ndvis   = [o.ndvi for o in in_win]
        smooth  = self._smooth(ndvis, self.settings.ndvi_smoothing_window)

        # ── Greenup detection ──────────────────────────────────────────────────
        thr = self.settings.ndvi_greenup_threshold
        greenup_date = greenup_ndvi = None
        for i in range(len(dates)):
            if smooth[i] > baseline + thr:
                if i + 1 < len(smooth) and smooth[i+1] > baseline + thr * 0.5:
                    greenup_date = dates[i]
                    greenup_ndvi = smooth[i]
                    break

        if greenup_date is None:
            return NDVIGreenupSignal(
                available=True, baseline_ndvi=round(baseline, 4),
                confidence=0.1,
                cloud_gap_days=self._max_gap(observations, search_start, win_end),
                n_observations=len(in_win),
                ndvi_timeseries=self._ts(in_win),
            )

        estimated_planting = greenup_date - timedelta(days=self.PLANTING_OFFSET)

        # ── Phenological profile ───────────────────────────────────────────────
        peak_idx  = smooth.index(max(smooth))
        peak_ndvi = round(smooth[peak_idx], 4)
        peak_date = dates[peak_idx]

        # Senescence: first date after peak where NDVI < 70% of peak
        senescence_date = None
        for i in range(peak_idx + 1, len(dates)):
            if smooth[i] < peak_ndvi * 0.70:
                senescence_date = dates[i]
                break

        season_length = (senescence_date - greenup_date).days if senescence_date else None

        # NDVI integral (trapezoidal, day-weighted)
        ndvi_integral = None
        if len(in_win) >= 2:
            integral = sum(
                (smooth[i] + smooth[i-1]) / 2.0 * (in_win[i].obs_date - in_win[i-1].obs_date).days
                for i in range(1, len(in_win))
            )
            ndvi_integral = round(integral, 2)

        # Growth rate: NDVI/day from planting to peak
        ndvi_rise_rate = None
        days_to_peak = (peak_date - greenup_date).days
        if days_to_peak > 0:
            ndvi_rise_rate = round((peak_ndvi - baseline) / days_to_peak, 5)

        # NDVI at ~90 days post-planting (harvest proxy)
        target_90d = estimated_planting + timedelta(days=90)
        closest    = min(in_win, key=lambda o: abs((o.obs_date - target_90d).days), default=None)
        ndvi_at_harvest = round(closest.ndvi, 4) if closest and abs((closest.obs_date - target_90d).days) < 20 else None

        gap  = self._max_gap(observations, search_start, win_end)
        conf = self._score(peak_ndvi - baseline, len(in_win), gap, estimated_planting, win_start, win_end)

        return NDVIGreenupSignal(
            greenup_date=greenup_date,
            estimated_planting_date=estimated_planting,
            planting_offset_days=self.PLANTING_OFFSET,
            baseline_ndvi=round(baseline, 4),
            peak_ndvi=peak_ndvi,
            peak_date=peak_date,
            ndvi_change=round(peak_ndvi - baseline, 4),
            senescence_date=senescence_date,
            season_length_days=season_length,
            ndvi_at_harvest=ndvi_at_harvest,
            ndvi_integral=ndvi_integral,
            ndvi_rise_rate=ndvi_rise_rate,
            ndvi_timeseries=self._ts(in_win),
            cloud_gap_days=gap,
            n_observations=len(in_win),
            confidence=conf,
            available=True,
        )

    def _ts(self, obs: list[NDVIObservation]) -> list[dict]:
        return [{"date": str(o.obs_date), "ndvi": o.ndvi, "evi": o.evi,
                 "ndwi": o.ndwi, "cloud_pct": o.cloud_cover_pct} for o in obs]

    def _smooth(self, values: list[float], window: int = 5) -> list[float]:
        n = len(values)
        if n < window:
            return values
        try:
            from scipy.signal import savgol_filter
            w = window if window % 2 == 1 else window - 1
            return list(savgol_filter(values, w, min(2, w - 1)))
        except ImportError:
            h = window // 2
            return [statistics.mean(values[max(0, i-h): min(n, i+h+1)]) for i in range(n)]

    def _max_gap(self, obs: list[NDVIObservation], start: date, end: date) -> int:
        dates = sorted(o.obs_date for o in obs if start <= o.obs_date <= end)
        if len(dates) < 2:
            return (end - start).days
        return max((dates[i+1] - dates[i]).days for i in range(len(dates) - 1))

    def _score(self, change, n_obs, gap, planting, win_start, win_end):
        # change: normalised at 0.15 — semi-arid/lower-midland EAK greenup deltas
        #         average 0.12–0.15; using 0.20 was capping scores at 0.60-0.75
        c = min(1.0, change / 0.15)
        # obs: 3 clear scenes is the realistic minimum for Kenya rainy-season S2;
        #      was 5, which penalised every small/cloudy farm by 0.60×
        o = min(1.0, n_obs / 3)
        # gap: 45-day normaliser — a 13d cloud gap is normal in Kenya rainy season
        g = max(0.4, 1.0 - gap / 45)
        w = 1.0 if win_start <= planting <= win_end else max(0.3, 1.0 - max(
            0, (win_start - planting).days, (planting - win_end).days) / 30)
        return round(c * o * g * w, 3)


# ── Signal 3: SAR Tillage + Phenological Features ─────────────────────────────

class SARTillageDetector:
    """
    Tillage: VV backscatter drops >= -1.5 dB from pre-season baseline
    Moisture: VH/VV cross-pol ratio increases post-rainfall

    Also extracts SAR phenological features for crop ID pipeline:
      vv_baseline, vv_at_peak_ndvi, vh_at_peak_ndvi,
      cross_pol_at_peak, full vv_timeseries
    """

    def __init__(self):
        self.settings = get_settings()

    def detect(
        self,
        observations: list[SARObservation],
        season_window: SeasonWindow,
        year: int,
        peak_ndvi_date: Optional[date] = None,   # passed in from NDVI signal
    ) -> SARTillageSignal:
        if not observations:
            return SARTillageSignal(available=False)

        win_start, win_end = season_window.get_window(year)
        search_start       = season_window.get_search_start(year)
        baseline_start     = search_start - timedelta(days=28)

        sar          = sorted(observations, key=lambda x: x.obs_date)
        baseline_obs = [o for o in sar if baseline_start <= o.obs_date < search_start]
        in_win       = [o for o in sar if search_start <= o.obs_date <= win_end]

        if len(in_win) < 2:
            return SARTillageSignal(available=True, confidence=0.1)

        baseline_vv = (statistics.mean([o.vv_db for o in baseline_obs]) if baseline_obs
                       else statistics.mean([o.vv_db for o in in_win[:2]]))

        # ── Tillage detection ──────────────────────────────────────────────────
        thr          = self.settings.sar_tillage_threshold_db
        tillage_date = vv_change = None
        moisture     = False

        for i, obs in enumerate(in_win):
            delta = obs.vv_db - baseline_vv
            if delta <= thr:
                tillage_date = obs.obs_date
                vv_change    = round(delta, 3)
                break
            if obs.cross_pol_ratio is not None:
                prev = [o.cross_pol_ratio for o in in_win[:i] if o.cross_pol_ratio is not None]
                if prev and obs.cross_pol_ratio > statistics.mean(prev) + 0.03:
                    moisture = True
                    if tillage_date is None:
                        tillage_date = obs.obs_date

        # ── SAR at NDVI peak (for crop ID) ────────────────────────────────────
        vv_at_peak = vh_at_peak = cross_at_peak = None
        if peak_ndvi_date and in_win:
            closest = min(in_win, key=lambda o: abs((o.obs_date - peak_ndvi_date).days))
            if abs((closest.obs_date - peak_ndvi_date).days) < 12:
                vv_at_peak    = closest.vv_db
                vh_at_peak    = closest.vh_db
                cross_at_peak = closest.cross_pol_ratio

        vv_ts = [{"date": str(o.obs_date), "vv_db": o.vv_db, "vh_db": o.vh_db,
                  "cross_pol": o.cross_pol_ratio, "pass": o.pass_direction} for o in in_win]

        if tillage_date is None and moisture:
            return SARTillageSignal(
                onset_date=in_win[0].obs_date, vv_baseline=round(baseline_vv, 3),
                vv_at_peak_ndvi=vv_at_peak, vh_at_peak_ndvi=vh_at_peak,
                cross_pol_at_peak=cross_at_peak, vv_timeseries=vv_ts,
                moisture_increase_detected=True, tillage_detected=False,
                confidence=0.35, available=True,
            )

        if tillage_date is None:
            return SARTillageSignal(
                available=True, confidence=0.15,
                vv_baseline=round(baseline_vv, 3), vv_timeseries=vv_ts,
            )

        conf = self._score(vv_change or 0.0, moisture, len(in_win), tillage_date, win_start, win_end)

        return SARTillageSignal(
            onset_date=tillage_date, vv_change_db=vv_change,
            vv_baseline=round(baseline_vv, 3),
            vv_at_peak_ndvi=vv_at_peak, vh_at_peak_ndvi=vh_at_peak,
            cross_pol_at_peak=cross_at_peak, vv_timeseries=vv_ts,
            moisture_increase_detected=moisture, tillage_detected=True,
            confidence=conf, available=True,
        )

    def _score(self, vv_change, moisture, n_obs, tillage_date, win_start, win_end):
        s = min(1.0, abs(vv_change) / 3.0)
        m = 1.1 if moisture else 1.0
        # S1 revisit ~6d; 4 passes per season is the realistic minimum (not 6)
        o = min(1.0, n_obs / 4)
        w = 1.0 if win_start <= tillage_date <= win_end else max(
            0.3, 1.0 - max(0, (win_start - tillage_date).days,
                           (tillage_date - win_end).days) / 30)
        return round(min(1.0, s * m * o * w), 3)


# ── Ensemble ───────────────────────────────────────────────────────────────────

class PlantingDateEnsemble:
    """
    Weighted combination using AEZ-specific weights (or global defaults).
    - Missing signal weights redistributed
    - Spread <= 7d → +15% confidence bonus
    - Spread > 21d → up to -40% penalty
    - Low confidence → blended with climatological onset
    """

    def __init__(self):
        self.settings = get_settings()

    def combine(
        self,
        rainfall: RainfallOnsetSignal,
        ndvi: NDVIGreenupSignal,
        sar: SARTillageSignal,
        season_window: SeasonWindow,
        year: int,
        fallback: bool = True,
        aez_weights: tuple[float, float, float] | None = None,
    ) -> tuple[Optional[date], float, str]:
        """Returns (estimated_date, confidence_score, method_label).

        aez_weights: optional (rainfall, ndvi, sar) override from AEZConfig.
        Falls back to global Settings weights if None.
        """
        s = self.settings
        w_rain = aez_weights[0] if aez_weights else s.weight_rainfall
        w_ndvi = aez_weights[1] if aez_weights else s.weight_ndvi
        w_sar  = aez_weights[2] if aez_weights else s.weight_sar

        signal_dates, weights, confs = [], [], []

        if rainfall.available and rainfall.onset_date:
            signal_dates.append(rainfall.onset_date)
            weights.append(w_rain)
            confs.append(rainfall.confidence)

        if ndvi.available and ndvi.estimated_planting_date:
            signal_dates.append(ndvi.estimated_planting_date)
            weights.append(w_ndvi)
            confs.append(ndvi.confidence)

        if sar.available and sar.onset_date:
            signal_dates.append(sar.onset_date)
            weights.append(w_sar)
            confs.append(sar.confidence)

        if not signal_dates:
            if fallback:
                return season_window.get_climatological_onset(year), 0.20, "fallback_climatology"
            return None, 0.0, "no_data"

        total_w  = sum(weights)
        norm_w   = [w / total_w for w in weights]
        ens_ord  = sum(d.toordinal() * w for d, w in zip(signal_dates, norm_w))
        ens_date = date.fromordinal(round(ens_ord))

        base_conf  = sum(c * w for c, w in zip(confs, norm_w))
        final_conf = min(1.0, base_conf * self._agreement(signal_dates))
        if len(signal_dates) == 1:
            # Single-signal detection: modest haircut (was 0.70, too harsh for
            # small farms that rarely trigger all three sensors simultaneously)
            final_conf *= 0.82

        n = len(signal_dates)
        if n == 3:
            method = "ensemble"
        elif n == 2:
            method = "ensemble_2signal"
        elif rainfall.available and rainfall.onset_date:
            method = "rainfall_only"
        elif ndvi.available and ndvi.estimated_planting_date:
            method = "ndvi_only"
        else:
            method = "sar_only"

        if final_conf < s.min_confidence_score and fallback:
            clim     = season_window.get_climatological_onset(year)
            ens_date = date.fromordinal(round(
                ens_date.toordinal() * 0.3 + clim.toordinal() * 0.7
            ))
            method      = f"{method}_clim_blend"
            final_conf  = max(final_conf, 0.25)

        return ens_date, round(final_conf, 3), method

    def _agreement(self, dates: list[date]) -> float:
        if len(dates) < 2:
            return 1.0
        spread = max(d.toordinal() for d in dates) - min(d.toordinal() for d in dates)
        # Planting-offset logic means NDVI greenup date is already shifted back
        # 12 days relative to rainfall/SAR onset — so 14-day spread is expected.
        if   spread <= 7:  return 1.15
        elif spread <= 14: return 1.05   # was 1.0 — small spread is a positive signal
        elif spread <= 21: return 0.95   # was 0.90
        elif spread <= 35: return 0.82   # was linear decay from 0.90 → 0.60
        else:              return max(0.70, 1.0 - (spread - 35) / 80)  # floor 0.70 not 0.60

    def confidence_level(self, conf: float) -> str:
        s = self.settings
        if   conf >= s.high_confidence_score: return "HIGH"
        elif conf >= 0.35:                    return "MEDIUM"  # was 0.40
        elif conf > s.min_confidence_score:   return "LOW"
        else:                                 return "UNCERTAIN"
