import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

MODEL_PATH = 'task_priority_model.pkl'
MIN_TRAINING_SAMPLES = 10

def get_model_pipeline():
    """Defines the machine learning pipeline."""
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    return model_pipeline

def train_model(app, db, Task):
    """Fetches tasks from the DB, retrains the model, and saves it."""
    print("Starting model training...")
    with app.app_context():
        training_tasks = Task.query.filter(Task.user_priority != None).all()
    
    if len(training_tasks) < MIN_TRAINING_SAMPLES:
        message = f"Not enough data to train. Need at least {MIN_TRAINING_SAMPLES} prioritized tasks. You have {len(training_tasks)}."
        print(message)
        return message

    X_train = [task.content for task in training_tasks]
    y_train = [task.user_priority for task in training_tasks]
    print(f"Training on {len(X_train)} samples.")

    model = get_model_pipeline()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    
    message = f"Model trained successfully on {len(X_train)} tasks and saved."
    print(message)
    return message

def predict_priority(task_content):
    """Loads the saved model and predicts priority for a new task."""
    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict([task_content])
        return prediction[0]
    except FileNotFoundError:
        print("Model file not found. Returning default priority (2=Medium).")
        return 2
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 2

def get_model_metrics(app, db, Task):
    """
    Evaluates the current model's performance on the existing training data.
    Returns a dictionary of metrics for the dashboard.
    """
    with app.app_context():
        training_tasks = Task.query.filter(Task.user_priority != None).all()
    
    if len(training_tasks) < MIN_TRAINING_SAMPLES:
        return {'accuracy': None, 'confusion_matrix': None, 'priority_counts': None}

    X_true_text = [task.content for task in training_tasks]
    y_true = [task.user_priority for task in training_tasks]

    try:
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X_true_text)
        
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
        priority_counts = Counter(y_true)

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'priority_counts': dict(priority_counts)
        }
    except FileNotFoundError:
        return {'accuracy': None, 'confusion_matrix': None, 'priority_counts': None}