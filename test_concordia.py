import concordia
import inspect
print("Concordia module loaded successfully.")
# List top-level attributes to find where GameMaster or similar might live
print(dir(concordia))
print("GameMaster class:", getattr(concordia, "GameMaster", None))

# Get the file path and structure—avoid None
print(concordia.__file__)
print(inspect.getsource(concordia))