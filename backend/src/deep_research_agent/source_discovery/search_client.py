from __future__ import annotations

from .contracts import SearchProviderConfig, SearchProviderResult, SearchQuery
from .providers import provider_from_config


class SearchClient:
    def __init__(self, config: SearchProviderConfig) -> None:
        self.config = config
        self.provider = provider_from_config(config)

    def search_many(
        self,
        queries: list[SearchQuery],
        *,
        max_candidates_per_query: int,
    ) -> list[SearchProviderResult]:
        return [
            self.provider.search(query, max_results=max_candidates_per_query) for query in queries
        ]
