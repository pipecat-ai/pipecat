# -*- coding: utf-8 -*-
class Credential:
    def __init__(self, secret_id, secret_key, token=""):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.token = token
