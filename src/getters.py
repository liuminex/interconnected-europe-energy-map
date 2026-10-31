import os
import pandas as pd

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.utils import get_abs_path

ZONES_DF = pd.read_csv(get_abs_path("../sample_data/zones.csv"), sep=";")

def get_zones_df():
    return ZONES_DF

def get_zone_dam_prices(zone_name: str) -> pd.DataFrame:

    file_path = get_abs_path(f"../sample_data/prices/{zone_name}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prices file for zone '{zone_name}' not found at {file_path}")
        
    df = pd.read_csv(file_path, sep=";")
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
    return df


def get_cap_limits_df(zone_from: str, zone_to: str, sample: bool = True) -> pd.DataFrame:

    file_path = get_abs_path(f"../sample_data/capacities/cap_limits_{zone_from}-{zone_to}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Capacity limits file for '{zone_from}-{zone_to}' not found at {file_path}")
        
    df = pd.read_csv(file_path, sep=";")
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
    return df


if __name__ == "__main__":

    zone_name = "DE_LU"
    zone_from = "DE_LU"
    zone_to = "FR"
    
    try:
        prices_df = get_zone_dam_prices(zone_name)
        print(f"DAM Prices for {zone_name}:\n", prices_df.head())
        
        cap_limits_df = get_cap_limits_df(zone_from, zone_to)
        print(f"\nCapacity Limits between {zone_from} and {zone_to}:\n", cap_limits_df.head())
        
        scheduled_exchange_series = get_scheduled_commercial_exchange(zone_from, zone_to)
        print(f"\nScheduled Commercial Exchange between {zone_from} and {zone_to}:\n", scheduled_exchange_series.head())
        
    except (FileNotFoundError, ValueError) as e:
        print(e)