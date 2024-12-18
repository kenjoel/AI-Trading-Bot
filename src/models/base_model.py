from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, filepath: str):
        pass

    @abstractmethod
    def load(self, filepath: str):
        pass
