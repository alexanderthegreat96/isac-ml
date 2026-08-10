from calc import IsacCalculator
import polars as pl

FILE_PATH = "isac-ml-training-dataset.csv"

# i just wanna see these
relevant_columns = [
    'username', 
    'timePlayed', 
    'weaponHits', 
    'headshots', 
    'bodyshots', 
    'headshotsPerHour', 
    'bodyshotsPerHour', 
    'weaponHitsPerHour'
]

df = pl.read_csv(FILE_PATH)
print("-- Original Data --")
print(df.select(relevant_columns))

rows = df.to_dicts()
updated_rows = []

for row in rows:
    time_played = row.get('timePlayed', 0)
    weapon_hits = row.get('weaponHits', 0)
    headshots = row.get('headshots', 0)
    
    # current data
    bodyshots = row.get('bodyshots', 0)
    hs_per_hour = row.get('headshotsPerHour', 0)
    bs_per_hour = row.get('bodyshotsPerHour', 0)
    wh_per_hour = row.get('weaponHitsPerHour', 0)
    
    # if either of thoes are missing recompute and update
    base_data_valid = (time_played != 0) and (weapon_hits != 0)
    any_is_zero = (
        bodyshots == 0 or bodyshots is None or 
        hs_per_hour == 0 or hs_per_hour is None or 
        bs_per_hour == 0 or bs_per_hour is None or 
        wh_per_hour == 0 or wh_per_hour is None
    )
    
    if base_data_valid and any_is_zero:
        calc = IsacCalculator(
            time_played_total=time_played,
            headshots=headshots,
            sum_hits=weapon_hits,
            kills_npc=row.get('npcKills', 0),
            kills_headshot=row.get('headshotKills', 0)
        )
        
        
        row['bodyshots'] = calc.bodyshots
        row['headshotsPerHour'] = calc.calculate_headshots_per_hour()
        row['bodyshotsPerHour'] = calc.calculate_bodyshots_per_hour()
        row['weaponHitsPerHour'] = calc.calculate_weapon_hits_per_hour()
        
    updated_rows.append(row)


updated_df = pl.DataFrame(updated_rows)
print("-- Updated Data --")
print(updated_df.select(relevant_columns))
updated_df.write_csv(FILE_PATH)
