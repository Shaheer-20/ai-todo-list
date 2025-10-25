import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from model import train_model, predict_priority

# --- App Configuration ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'
app.config['SECRET_KEY'] = 'a_really_secret_key_change_this' # Change this!
db = SQLAlchemy(app)

# --- Database Model ---
# We will store priority as integers: 1=Low, 2=Medium, 3=High
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(300), nullable=False)
    
    # The model's prediction
    suggested_priority = db.Column(db.Integer, default=2) 
    
    # The user's final decision (this is our "ground truth" for training)
    user_priority = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'<Task {self.id}: {self.content}>'

# --- Utility Function ---
def get_priority_name(priority_int):
    """Converts a priority number (1, 2, 3) to a name."""
    if priority_int == 3:
        return "High"
    if priority_int == 1:
        return "Low"
    return "Medium" # Default

# --- Main App Routes ---
@app.route('/')
def index():
    """
    Main page: Displays all tasks, ordered by their priority.
    Tasks with a user-set priority come first.
    """
    tasks = Task.query.order_by(Task.user_priority.desc(), Task.suggested_priority.desc()).all()
    
    # Pass the helper function to the template
    return render_template('index.html', tasks=tasks, get_priority_name=get_priority_name)

@app.route('/add', methods=['POST'])
def add_task():
    """
    Handles adding a new task.
    It predicts the priority before saving.
    """
    task_content = request.form['content']
    
    if not task_content:
        flash("Task content cannot be empty!", "error")
        return redirect(url_for('index'))

    # --- Machine Learning Integration ---
    # 1. Predict the priority of the new task content
    predicted_p = predict_priority(task_content)
    # ------------------------------------

    # 2. Create the new task with the *suggested* priority
    new_task = Task(content=task_content, suggested_priority=int(predicted_p))
    
    try:
        db.session.add(new_task)
        db.session.commit()
        flash(f"Task added with a suggested priority of '{get_priority_name(predicted_p)}'", "success")
    except Exception as e:
        flash(f"Error adding task: {e}", "error")
        
    return redirect(url_for('index'))

@app.route('/set_priority/<int:id>', methods=['POST'])
def set_priority(id):
    """
    Handles when a user *manually* sets a task's priority.
    This is the "learning" step, creating a training example.
    """
    task = Task.query.get_or_404(id)
    
    # Get the priority from the form button (e.g., <button name="priority" value="3">High</button>)
    new_priority = request.form['priority']
    
    # This is the "label" (y) our model will learn from
    task.user_priority = int(new_priority)
    
    try:
        db.session.commit()
        flash("Task priority updated!", "info")
    except Exception as e:
        flash(f"Error setting priority: {e}", "error")
        
    return redirect(url_for('index'))

@app.route('/delete/<int:id>', methods=['POST'])
def delete_task(id):
    """Deletes a task from the database."""
    task_to_delete = Task.query.get_or_404(id)
    
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        flash("Task deleted.", "info")
    except Exception as e:
        flash(f"Error deleting task: {e}", "error")
        
    return redirect(url_for('index'))

# --- Model Training Route ---
@app.route('/retrain-model')
def trigger_training():
    """
    A special (un-linked) route to trigger the model retraining.
    In a real app, you would secure this or run it on a schedule.
    """
    try:
        message = train_model()
        flash(message, "success")
    except Exception as e:
        flash(f"Error during training: {e}", "error")
        
    return redirect(url_for('index'))

# --- Main Entry Point ---
if __name__ == "__main__":
    # Create the database file if it doesn't exist
    if not os.path.exists('todos.db'):
        with app.app_context():
            db.create_all()
        print("Database created!")
        
    app.run(debug=True)