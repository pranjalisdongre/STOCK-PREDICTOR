import pandas as pd
import sqlite3
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    def __init__(self, db_path='stock_data.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with
        