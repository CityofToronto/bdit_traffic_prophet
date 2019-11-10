# TO DO - move this to a yaml file.  Eventually will be handled by Kedro.
# https://martin-thoma.com/configuration-files-in-python/
# https://kedro.readthedocs.io/en/latest/04_user_guide/03_configuration.html

# For countmatch
cm = {
    'min_stn_count': 96,
    'min_permanent_stn_days': 274,
    'exclude_ptc_neg': [8540609, 446378, 12336151, 5439677, 1145406, 30019302,
                        7094867, 9722624, 439225, 1146926, 1141002, 440202,
                        1147135],
    'exclude_ptc_pos': [446402, 7204532, 1145377, 30029635, 1147358, 106853,
                        1140996, 1797, 14177830, 30073989, 14189397,
                        440428, 14659261]
}

distances = {
    # Lat-lon of 703 Don Mills.
    'centre_of_toronto': [-79.333536, 43.708975],
    # Counterclockwise angle from due north for Toronto's street grid
    # (measured from Spadina/Lakeshore to Spadina/Dupont), in degres.
    'toronto_street_angle': 16.485518084102
}

# DO NOT STORE PERSONAL CREDENTIALS HERE!
postgres = {
    # Name of database.
    'database': 'bigdata',
    # Name of schema and table.
    'schema_table': 'prj_volume.uoft_centreline_volumes_output'
}
