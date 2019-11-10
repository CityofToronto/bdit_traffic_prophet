"""Base classes for connecting to Postgres database."""

import psycopg2
import configparser

from . import cfg


class Connection:
    # TO DO: this currently has NO FORMAL TEST SUITE, mainly because we'll
    # be replacing it with whatever Kedro cooked up at some point.

    def __init__(self, credential_file, credential_name):
        config = configparser.RawConfigParser()
        config.read(credential_file.as_posix())
        self.credentials = dict(config[credential_name])
        self.dbname = cfg.postgres['database']

    def connect(self):
        return psycopg2.connect(database=self.dbname, **self.credentials)
