from torchcast.kalman_filter import KalmanFilter
from torchcast.process import Season, LocalTrend, LocalLevel


class CTAModel(KalmanFilter):
    measure_col_name = 'rides'

    def __init__(self):
        super().__init__(
            measures=[self.measure_col_name],
            processes=[
                # seasonal processes:
                Season(id='day_in_week', period=24*7, dt_unit='h', K=3, fixed=True),
                Season(id='day_in_year', period=24*365.25, dt_unit='h', K=8, fixed=True),
                # long-running trend:
                LocalTrend(id='trend'),
                # 'local' processes: allow temporary deviations that decay
                LocalLevel(id='level', decay=True)
            ]
        )

    def fit(self, ds):
        return super().fit(
            ds.tensors[0],
            start_offsets=ds.start_datetimes
        )
