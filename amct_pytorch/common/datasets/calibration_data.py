from .base_provider import BaseDataProvider

class CalibrationDataProvider(BaseDataProvider):
    def __init__(self, data_source, config=None):
        self.data_source = data_source
        self.config = config or {}

    def __iter__(self):
        return iter(self.data_source)

    def get_batch(self, batch_size: int):
        batch = []
        for i, item in enumerate(self.data_source):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def get_sample_count(self) -> int:
        return len(self.data_source)