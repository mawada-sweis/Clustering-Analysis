def get_feature_types_stats(df):
    features = df.dtypes.rename('feature_type').reset_index().rename(columns={'index':'feature'})
    return features 


def get_feature_missing(df): 
    df_rows = df.shape[0]
    missing_stats = df.isna().sum().rename('missing_count').reset_index()
    dont_know = df[df == 998].notna().sum().rename('dont_know_count').reset_index()
    no_response = df[df == -9998].notna().sum().rename('no_response_count').reset_index()
    technological_error = df[df == 9999].notna().sum().rename('technological_error_count').reset_index()
    prefer_not_to_answer = df[df == 999].notna().sum().rename('prefer_not_to_answer_count').reset_index()
    not_required = df[df == 995].notna().sum().rename('not_required_count').reset_index()
    
    missing_stats['missing_percentage'] = missing_stats['missing_count']/df_rows
    missing_stats['dont_know_percentage'] = dont_know['dont_know_count']/df_rows
    missing_stats['no_response_percentage'] = no_response['no_response_count']/df_rows
    missing_stats['technological_error_percentage'] = technological_error['technological_error_count']/df_rows 
    missing_stats['not_required_percentage'] = not_required['not_required_count']/df_rows
    missing_stats['prefer_not_to_answer_percentage'] = prefer_not_to_answer['prefer_not_to_answer_count'] /df_rows
    
    feature_types_df = get_feature_types_stats(df)
    
    missing_stats = missing_stats.rename(columns={'index':'feature'}).merge(feature_types_df,
                         left_on='feature',
                         right_on='feature',
                         how='left')
    missing_stats.sort_values('missing_percentage', ascending=False)
    return missing_stats 
