'''
src/cleaner.py
Production-grade vectorized telemetry data cleaner.
Applies bounded clipping, group-aware imputation, and vectorized KPI calculations.
'''

from pathlib import Path
import pandas as pd
import numpy as np

class TelemetryCleaner:
    #  Cleans raw industrial telemetry stream using vectorized operations
    def __init__(self, raw_filepath: str='data/raw/raw_telemetry.csv'):
        self.raw_filepath = Path(raw_filepath)
        self.df: pd.DataFrame = pd.DataFrame()
        
    def load_data(self) -> 'TelemetryCleaner':
        '''Loads raw CSV and parses ISO timestamps'''
        if not self.raw_filepath.exists():
            raise FileNotFoundError(f"Input file not found at: {self.raw_filepath}")
        
        self.df = pd.read_csv(self.raw_filepath, parse_dates=['timestamp'])
        #  Ensure data is sorted by machine and chronological order
        self.df = self.df.sort_values(by=['machine_id', 'timestamp']).reset_index(drop=True)
        return self
    
    def clean_rpm(self) -> 'TelemetryCleaner':
        '''Corrects negative sensor readings using Numpy Vectorizations
        RPM can't be negative in unidirectional looms/spinning drives
        '''
        # Vectorized absolute value on negative values
        self.df['rpm'] = np.where(self.df['rpm'] < 0, np.abs(self.df['rpm']), self.df['rpm'])
        return self
    def clean_power(self, max_threshold_kw: float = 25.0) -> TelemetryCleaner:
        '''
        Imputes missing power drops and clips unrealistic high-voltage spikes.
        Uses group-aware forward/backward filling per machine.
        '''
        
        self.df['power_kw'] = (
            self.df.groupby('machine_id')['power_kw']
            .ffill().bfill()        
        )
        # 2. Vectorized clipping: Cap transient surge spikes above equipment threshold
        self.df["power_kw"] = np.clip(self.df["power_kw"], a_min=0.0, a_max=max_threshold_kw)

        return self
    
    def add_derived_metrics(self) -> 'TelemetryCleaner':
        # Computes incremental production and energy consumption per 10-sec interval
        # Compute incremental units produced per interval per machine
        #  diff() yields NaN on the first row of each group; fill NaN with zero
        self.df['units_delta'] = (
            self.df.groupby('machine_id')['production_counter']
            .diff()
            .fillna(0)
            .clip(lower=0)
        )
        
        # Energy consumed in kWh during this 10-sec interval
        # Energy (kWH) = Power(kW) * Time(hrs)
        #  10 sec = 10/3600 hrs
        time_delta_hours = 10.0/3600.0
        self.df['energy_kwh'] = np.round(self.df['power_kw'] * time_delta_hours, 5)
        
        # Efficiency: kwh per 100 units (avoid division by 0 during np.where)
        self.df['kwh_per_100_units'] = np.where(
            self.df['units_delta'] > 0,
            np.round((self.df['energy_kwh']/self.df['units_delta']) * 100, 4),
            np.nan
        )
        
        return self
    
    def export(self, output_path: str = 'data/processed/clean_telemetry.parquet') -> pd.DataFrame:
        '''
        Exports the cleaned dataset to columnar parquet format
        Parquet provides faster read speeds and smaller footprints than CSV
        '''
        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        #  Downcast string mahcine_id and status to categorical optimize RAM
        self.df['machine_id'] = self.df['machine_id'].astype('category')
        self.df['status'] = self.df['status'].astype('category')
        
        # Save as Parquet with snappy compression (fallback to CSV if pyarrow not installed)
        try:
            self.df.to_parquet(out_file, index=False)
            print(f"Exported Cleaned telemtry -> {out_file.resolve()} (Parquet)")
        except ImportError:
            fallback_csv = out_file.with_suffix('.csv')
            self.df.to_csv(fallback_csv, index=False)
            print(f"pyarrow missing. Exported fallback CSV -> {fallback_csv.resolve()}")
            
        return self.df
    
if __name__ == '__main__':
    cleaner = (
    TelemetryCleaner()
    .load_data()
    .clean_rpm()
    .clean_power()
    .add_derived_metrics()
    )

df_clean = cleaner.export()

print('\nCleaned Data Sample: ')
print(df_clean[["timestamp", "machine_id", "status", "rpm", "power_kw", "units_delta", "energy_kwh"]].head(10))

print('\nNull Value Audit Post-Cleaning: ')
print(df_clean[['rpm', 'power_kw', 'energy_kwh']].isnull().sum())