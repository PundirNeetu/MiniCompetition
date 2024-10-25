
from sklearn.model_selection import train_test_split

def split(X, y):
    """
    Train validation split.

    Parameters:
    X, y

    Returns:
    train_X, val_X, train_y, val_y
    """
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42,test_size=0.2)
    return train_X, val_X, train_y, val_y