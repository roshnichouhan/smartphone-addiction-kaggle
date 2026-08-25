import pandas as pd


def create_features(df):
    """
    Create new features from the original smartphone addiction data.
    """

    df = df.copy()

    # ==========================================
    # 1. Entertainment hours
    # Social media + Gaming
    # ==========================================

    df["entertainment_hours"] = (
        df["social_media_hours"] +
        df["gaming_hours"]
    )

    # ==========================================
    # 2. Screen time / Sleep ratio
    # ==========================================

    df["screen_sleep_ratio"] = (
        df["daily_screen_time_hours"] /
        (df["sleep_hours"] + 1)
    )

    # ==========================================
    # 3. Weekend screen / Daily screen ratio
    # ==========================================

    df["weekend_daily_ratio"] = (
        df["weekend_screen_time"] /
        (df["daily_screen_time_hours"] + 1)
    )

    # ==========================================
    # 4. Digital activity
    # Notifications + App opens
    # ==========================================

    df["digital_activity"] = (
        df["notifications_per_day"] +
        df["app_opens_per_day"]
    )

    # ==========================================
    # 5. Total phone activity hours
    # ==========================================

    df["phone_activity_hours"] = (
        df["daily_screen_time_hours"] +
        df["social_media_hours"] +
        df["gaming_hours"]
    )

    return df