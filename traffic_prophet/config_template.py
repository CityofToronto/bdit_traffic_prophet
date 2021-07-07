# TO DO - move this to a yaml file.  Eventually will be handled by Kedro.
# https://martin-thoma.com/configuration-files-in-python/
# https://kedro.readthedocs.io/en/latest/04_user_guide/03_configuration.html

# For countmatch.
cm = {
    'verbose': False,
    'min_year': 2006,
    'max_year': 2018,
    'min_count': 96,
    'min_counts_in_day': 24,
    'min_permanent_months': 12,
    'min_permanent_days': 274,
    'exclude_ptc_neg': [8540609, 446378, 12336151, 5439677, 1145406, 30019302,
                        7094867, 9722624, 439225, 1146926, 1141002, 440202,
                        1147135],
    'exclude_ptc_pos': [446402, 7204532, 1145377, 30029635, 1147358, 106853,
                        1140996, 1797, 14177830, 30073989, 14189397,
                        440428, 14659261],
    'derived_vals_calculator': 'Standard',
    'derived_vals_settings': {},
    'growth_factor_calculator': 'Composite',
    'growth_factor_settings': {},
    'n_neighbours': 5,
    'match_single_direction': True,
    'average_growth': True,
    'matcher': 'Standard',
    'matcher_settings': {}
}

# For calculating nearest neighbours in countmatch.neighbour.
distances = {
    # Lat-lon of 703 Don Mills.
    'centre_of_toronto': [-79.333536, 43.708975],
    # Counterclockwise angle from due north for Toronto's street grid
    # (measured from Spadina/Lakeshore to Spadina/Dupont), in degres.
    'toronto_street_angle': 16.485518084102
}
