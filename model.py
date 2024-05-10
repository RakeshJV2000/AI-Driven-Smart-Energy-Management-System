def process_and_model_energy_data(df):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from io import StringIO

    # Load data
    data = df

    buffer = StringIO()
    df.head(10).to_html(buf=buffer)
    head_html = buffer.getvalue()

    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    buffer = StringIO()
    df.describe().to_html(buf=buffer)
    describe_html = buffer.getvalue()
   

    # Assume that 'static' is your Flask app's directory for static files
    # Drop unnecessary columns
    dropped_cols = ["net_manager", "purchase_area", "street", "zipcode_from", "zipcode_to", "city"]
    data = data.drop(dropped_cols, axis=1)

    # Handle missing values (if any)
    data = data.dropna()

    # Label encoding
    label_encoder = LabelEncoder()
    data['type_of_connection'] = label_encoder.fit_transform(data['type_of_connection'])

    # Features and target
    X = data.drop('annual_consume', axis=1)
    y = data['annual_consume']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Random Forest Regressor
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = random_forest_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {
        'Mean_Squared_Error': mse,
        'Mean_Absolute_Error': mae,
        'Head': head_html,
        'Info': info_str,
        'Describe': describe_html
    }