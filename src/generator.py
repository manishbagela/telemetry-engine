from pathlib import Path
import numpy as np
import pandas as pd

def generate_telemetry_data(
    num_records: int = 10_000,
    seed: int = 42,
    output_path: str = 'data/raw/raw_telemetry.csv',
) -> pd.DataFrame:
    '''
    Generates synthetic telemetry data for an industrial plant fleet
    
    Parameters:
    
    num_records: int
        Number of telemetry snapshot rows to generate
    seed: int
        Reproducibility seed for Numpy's Generator
    output_path: str
        Relative path to export the resulting CSV
        
    Returns:
    
    pd.DataFrame : The synthesized raw dataset
    '''
    
    rng = np.random.default_rng(seed)
    
    
    #1 : Fleet IDs
    machine_pool = np.array(['LOOM-01', 'LOOM-02', 'LOOM-03', 'SPIN-01', 'SPIN-02' ])
    machine_ids = rng.choice(machine_pool, size = num_records)
    
    # Base timestamp sequence: 10-second intervals starting from a fixed anchor
    start_time = np.datetime64('2026-11-01T06:00:00')
    timestamps = start_time + np.arange(num_records)*np.timedelta64(10, 's')
    
    # 2. Operating status
    #  Machine states: 0: stopped/off, 1: Idle, 2:  running (80% running probability)
    
    statuses = rng.choice(
        ['RUNNING', 'IDLE', 'STOPPED'],
        size = num_records,
        p = [0.8, 0.15, 0.05]
    )
    
    # 3. Vectorised Sensor generation (Normal Distributions)
    # Loom/SPin speeds: nominal mean = 650 rpm, std = 25 rpm
    rpm = rng.normal(loc=650.0, scale=25.0, size=num_records)
    
    # Power draw: nominal mean = 5.5 kW, std = 0.4 kW
    power_kw = rng.normal(loc=5.5, scale=0.4, size=num_records)
    
    #  Unit production counter: increments based on operating status
    #  Base Rate: ~12.5 picks/units per 10s tick when running
    units_tick = np.where(statuses == "RUNNING", rng.poisson(lam=12, size=num_records), 0)
    cumulative_units = np.cumsum(units_tick)
    
    #  Zero out RPM and power for machines that are stopped or idle
    rpm = np.where(statuses == 'STOPPED', 0.0, rpm)
    power_kw = np.where(statuses == 'STOPPED', 0.2, power_kw)
    
    rpm = np.where(statuses == 'IDLE', 50.0, rpm)
    power_kw = np.where(statuses == 'IDLE', 1.1, power_kw)
    
    '''
    4. INJECT REAL WORLD SENSOR CORRUPTIONS (NOISE & GLITCHES)
    '''
    
    # Glitch A: Dropped Telemetry packets (Missing/ NaN values in ~3% of power reading)
    
    nan_mask = rng.random(size=num_records) < 0.03
    power_kw[nan_mask] = np.nan
    
    #  Glitch B: Polarity/Sensor GLitch (Negative RPM readings in ~1% of records)
    negative_rpm_mask = rng.random(size=num_records) < 0.01
    rpm[negative_rpm_mask] = -1*np.abs(rpm[negative_rpm_mask])
    
    #  Glitch C: Sudden transient power spikes 
    
    spike_mask = rng.random(size=num_records) < 0.005
    power_kw[spike_mask] = power_kw[spike_mask] * rng.uniform(8.0, 12.0, size=np.sum(spike_mask))
    
    #  5. Assemple into a Pandas Dataframe 
    df = pd.DataFrame({
        'timeestamp' : timestamps,
        'machine_id' : machine_ids,
        'status' : statuses,
        'rpm' : np.round(rpm, 2),
        'power_kw' : np.round(power_kw, 3),
        'production_counter': cumulative_units
    })
    
    #  Ensure output directory exists and write CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated {num_records} telemetry rows -> {out_file.resolve()}")
    return df


if __name__ == '__main__':
    df = generate_telemetry_data(num_records=10000)
    print('\nDataset Snapshot: ')
    print(df.head(10))
    print('\nData Types & Info:')
    print(df.info())
    
    