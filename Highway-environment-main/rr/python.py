import highway_env
import gymnasium as gym

# 嘗試創建 Highway 環境
try:
    env = gym.make('highway-v0')
    print("Highway environment 'highway-v0' loaded successfully.")
except gym.error.NameNotFound:
    print("Highway environment 'highway-v0' is not available.")
