import os
import random
import time

"""
from algoliasearch.configs import SearchConfig
from algoliasearch.search_client import SearchClient
"""

class SearchIndexer():
    def __init__(self, story_id):
        pass

    def index_text(self, text):
        pass

    def index_image(self, text):
        pass
"""
class AlgoliaSearchIndexer(SearchIndexer):
    def __init__(self, story_id):
        self.index = None
        self.story_id = story_id

        self.search_enabled = os.getenv('ALGOLIA_APP_ID') and os.getenv('ALGOLIA_API_KEY')
        if self.search_enabled:
            config = SearchConfig(os.getenv('ALGOLIA_APP_ID'), os.getenv('ALGOLIA_API_KEY'))
            self.algolia = SearchClient.create_with_config(config)
            self.index = self.algolia.init_index('daily-llm-conversations')

    def index_text(self, text):
        if self.index:
            res = self.index.save_object({
                "objectID": hex(random.getrandbits(128))[2:],
                "storyID": self.story_id,
                "type": "text",
                "text": text,
                "createdAt": int(time.time())
            }).wait()

    def index_image(self, url):
        if self.index:
            self.index.save_object({
                "objectID": hex(random.getrandbits(128))[2:],
                "storyID": self.story_id,
                "type": "image",
                "image": url,
                "createdAt": int(time.time())
            }).wait()
"""
