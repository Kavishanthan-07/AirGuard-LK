from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
META.mkdir(parents=True, exist_ok=True)

# ============================================================
# SENSOR DICTIONARY
# ============================================================

sensors = [
    {
        "sensor": "TP2",
        "signal_type": "analogue",
        "unit": "bar",
        "group": "pressure",
        "meaning": "Pressure measured at the compressor.",
        "airguard_role": "Compressor pressure behaviour"
    },
    {
        "sensor": "TP3",
        "signal_type": "analogue",
        "unit": "bar",
        "group": "pressure",
        "meaning": "Pressure generated at the pneumatic panel.",
        "airguard_role": "System/pneumatic pressure behaviour"
    },
    {
        "sensor": "H1",
        "signal_type": "analogue",
        "unit": "bar",
        "group": "pressure",
        "meaning": "Pressure associated with the cyclonic separator filter discharge.",
        "airguard_role": "Separator/discharge pressure behaviour"
    },
    {
        "sensor": "DV_pressure",
        "signal_type": "analogue",
        "unit": "bar",
        "group": "pressure",
        "meaning": "Pressure drop associated with air-dryer tower discharge.",
        "airguard_role": "Dryer/compressor loading behaviour"
    },
    {
        "sensor": "Reservoirs",
        "signal_type": "analogue",
        "unit": "bar",
        "group": "pressure",
        "meaning": "Downstream reservoir pressure; normally close to TP3.",
        "airguard_role": "Stored-air pressure and pressure recovery"
    },
    {
        "sensor": "Oil_temperature",
        "signal_type": "analogue",
        "unit": "degC",
        "group": "temperature",
        "meaning": "Compressor oil temperature.",
        "airguard_role": "Thermal operating condition"
    },
    {
        "sensor": "Motor_current",
        "signal_type": "analogue",
        "unit": "A",
        "group": "electrical",
        "meaning": "Current measured from one phase of the compressor motor.",
        "airguard_role": "Motor loading and compressor activity"
    },
    {
        "sensor": "COMP",
        "signal_type": "digital",
        "unit": "binary",
        "group": "control",
        "meaning": "Air-intake-valve electrical signal.",
        "airguard_role": "Compressor operating-state context"
    },
    {
        "sensor": "DV_eletric",
        "signal_type": "digital",
        "unit": "binary",
        "group": "control",
        "meaning": "Electrical signal controlling the compressor outlet valve.",
        "airguard_role": "Loaded/offloaded state context"
    },
    {
        "sensor": "Towers",
        "signal_type": "digital",
        "unit": "binary",
        "group": "control",
        "meaning": "Indicates which air-drying tower is operating.",
        "airguard_role": "Dryer operating-state context"
    },
    {
        "sensor": "MPG",
        "signal_type": "digital",
        "unit": "binary",
        "group": "control",
        "meaning": "Control signal involved in starting compressor operation under load.",
        "airguard_role": "Compression-cycle context"
    },
    {
        "sensor": "LPS",
        "signal_type": "digital",
        "unit": "binary",
        "group": "protection",
        "meaning": "Low-pressure signal that activates below approximately 7 bar.",
        "airguard_role": "Low-pressure condition"
    },
    {
        "sensor": "Pressure_switch",
        "signal_type": "digital",
        "unit": "binary",
        "group": "control",
        "meaning": "Signal associated with discharge of the air-drying towers.",
        "airguard_role": "Dryer discharge state"
    },
    {
        "sensor": "Oil_level",
        "signal_type": "digital",
        "unit": "binary",
        "group": "protection",
        "meaning": "Signal that activates when compressor oil is below the expected level.",
        "airguard_role": "Oil-level condition"
    },
    {
        "sensor": "Caudal_impulses",
        "signal_type": "digital",
        "unit": "binary/pulse",
        "group": "flow",
        "meaning": "Pulse signal associated with air flow from the APU to the reservoirs.",
        "airguard_role": "Air-flow activity"
    },
]

sensor_df = pd.DataFrame(sensors)

sensor_df.to_csv(
    META / "metropt3_sensor_dictionary.csv",
    index=False
)

# ============================================================
# DOCUMENTED FAILURE EVENTS
# ============================================================

events = [
    {
        "event_id": "F01",
        "start_time": "2020-04-18 00:00:00",
        "end_time": "2020-04-18 23:59:00",
        "failure_type": "Air Leak",
        "severity": "High stress",
    },
    {
        "event_id": "F02",
        "start_time": "2020-05-29 23:30:00",
        "end_time": "2020-05-30 06:00:00",
        "failure_type": "Air Leak",
        "severity": "High stress",
    },
    {
        "event_id": "F03",
        "start_time": "2020-06-05 10:00:00",
        "end_time": "2020-06-07 14:30:00",
        "failure_type": "Air Leak",
        "severity": "High stress",
    },
    {
        "event_id": "F04",
        "start_time": "2020-07-15 14:30:00",
        "end_time": "2020-07-15 19:00:00",
        "failure_type": "Air Leak",
        "severity": "High stress",
    },
]

events_df = pd.DataFrame(events)

events_df.to_csv(
    META / "metropt3_failure_events.csv",
    index=False
)

print("Created:")
print(META / "metropt3_sensor_dictionary.csv")
print(META / "metropt3_failure_events.csv")
