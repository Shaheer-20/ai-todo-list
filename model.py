import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = 'task_priority_model.pkl'
MIN_TRAINING_SAMPLES = 10 # Minimum number of tasks to train a model

def get_model_pipeline():
    """
    Defines the machine learning pipeline.
    It converts text into numbers (TfidfVectorizer) and then classifies (MultinomialNB).
    """
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            stop_words='english', # Ignore common words (e.g., 'the', 'is', 'a')
            ngram_range=(1, 2)    # Look at single words and two-word phrases
        )),
        ('classifier', MultinomialNB())
    ])
    return model_pipeline

def train_model():
    """
    Fetches all user-prioritized tasks from the DB and retrains the model.
    Saves the trained model to a file using joblib.
    """
    print("Starting model training...")

    # Import here to avoid circular dependency
    from app import app, Task
    
    # Use app_context to allow database access outside of a Flask route
    with app.app_context():
        # 1. Get training data from the database
        # We only train on tasks the user has *explicitly* prioritized
        training_tasks = Task.query.filter(Task.user_priority != None).all()
    
    if len(training_tasks) < MIN_TRAINING_SAMPLES:
        message = f"Not enough data to train. Need at least {MIN_TRAINING_SAMPLES} prioritized tasks. You have {len(training_tasks)}."
        print(message)
        return message

    # 2. Format data for scikit-learn
    # X is the "feature" (the task content)
    # y is the "label" (the user's priority)
    X_train = [task.content for task in training_tasks]
    y_train = [task.user_priority for task in training_tasks]
    print(f"Training on {len(X_train)} samples.")

    # 3. Create and train the model
    model = get_model_pipeline()
    model.fit(X_train, y_train)

    # 4. Save the trained model to a file
    joblib.dump(model, MODEL_PATH)
    
    message = f"Model trained successfully on {len(X_train)} tasks and saved."
    print(message)
    return message

def predict_priority(task_content):
    """
    Loads the saved model and predicts the priority for a new, single task.
    """
    try:
        # Load the pre-trained model from disk
        model = joblib.load(MODEL_PATH)
        
        # .predict() expects a list or iterable, not a single string
        prediction = model.predict([task_content])
        
        # Return the first (and only) prediction
        return prediction[0]
        
    except FileNotFoundError:
        print("Model file not found. Returning default priority (2=Medium).")
        return 2 # Default to Medium priority if no model exists
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 2 # Default to Medium