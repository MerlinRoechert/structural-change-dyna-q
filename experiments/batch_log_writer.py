from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor


class BatchLogWriter:
    def __init__(self, result_processor: "ResultProcessor"):
        self.result_processor = result_processor
        self.record_count = 0
        self._queries: list[tuple[str, tuple]] = []

    def add_records(
        self,
        logtable_identifier: str,
        records: Iterable[dict],
    ) -> None:
        statement: str | None = None
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for record in records:
            if statement is None:
                table_name = (
                    f"{self.result_processor.database_config.table_name}"
                    f"__{logtable_identifier}"
                )
                columns = (
                    "experiment_id",
                    "timestamp",
                    *record.keys(),
                )
                statement = (
                    self.result_processor.db_connector.prepare_write_query(
                        table_name,
                        columns,
                    )
                )

            values = (
                self.result_processor.experiment_id,
                timestamp,
                *record.values(),
            )
            self._queries.append((statement, values))
            self.record_count += 1

    def write(self) -> None:
        self.result_processor.db_connector.execute_queries(self._queries)
