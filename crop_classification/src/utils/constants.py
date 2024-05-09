rabi_season_fns = ['oct_1f', 'oct_2f', 'nov_1f', 'nov_2f', 'dec_1f', 'dec_2f', 'jan_1f', 'jan_2f', 
                        'feb_1f', 'feb_2f', 'mar_1f', 'mar_2f', 'apr_1f', 'apr_2f']
kharif_season_fns = ['may_1f', 'may_2f', 'jun_1f', 'jun_2f', 'jul_1f', 'jul_2f', 'aug_1f', 'aug_2f', 'sep_1f', 'sep_2f',
                 'oct_1f', 'oct_2f', 'nov_1f', 'nov_2f']
season_fn_map = {'rabi':rabi_season_fns, 'kharif':kharif_season_fns}
fns_dates_map = {
    'oct_1f': ['10-01', '10-16'],
    'oct_2f': ['10-16', '11-01'],
    'nov_1f': ['11-01', '11-16'],
    'nov_2f': ['11-16', '12-01'],
    'dec_1f': ['12-01', '12-16'],
    'dec_2f': ['12-16', '01-01'],
    'jan_1f': ['01-01', '01-16'],
    'jan_2f': ['01-16', '02-01'],
    'feb_1f': ['02-01', '02-16'],
    'feb_2f': ['02-16', '03-01'],
    'mar_1f': ['03-01', '03-16'],
    'mar_2f': ['03-16', '04-01'],
    'apr_1f': ['04-01', '04-16'],
    'apr_2f': ['04-16', '05-01']
}
crop_label_map = {'0':'Mustard', '1':'Wheat', '2':'Potato'}