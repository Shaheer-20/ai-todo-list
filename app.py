import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import json
from model import train_model, predict_priority, get_model_metrics

# --- App Configuration ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'
app.config['SECRET_KEY'] = 'a_really_secret_key_change_this'
db = SQLAlchemy(app)

# --- Database Model ---
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(300), nullable=False)
    suggested_priority = db.Column(db.Integer, default=2)
    user_priority = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'<Task {self.id}: {self.content}>'

# --- Utility Function ---
def get_priority_name(priority_int):
    if priority_int == 3: return "High"
    if priority_int == 1: return "Low"
    return "Medium"

# --- Main App Routes ---
@app.route('/')
def index():
    tasks = Task.query.order_by(Task.user_priority.desc(), Task.suggested_priority.desc()).all()
    return render_template('index.html', tasks=tasks, get_priority_name=get_priority_name)

@app.route('/dashboard')
def dashboard():
    """Displays the model performance dashboard."""
    metrics = get_model_metrics(app, db, Task)
    metrics_json = json.dumps(metrics)
    return render_template('dashboard.html', metrics=metrics, metrics_json=metrics_json)

@app.route('/add', methods=['POST'])
def add_task():
    task_content = request.form['content']
    if not task_content:
        flash("Task content cannot be empty!", "error")
        return redirect(url_for('index'))

    predicted_p = predict_priority(task_content)
    new_task = Task(content=task_content, suggested_priority=int(predicted_p))
    db.session.add(new_task)
    db.session.commit()
    flash(f"Task added with a suggested priority of '{get_priority_name(predicted_p)}'", "success")
    return redirect(url_for('index'))

@app.route('/edit/<int:id>', methods=['POST'])
def edit_task(id):
    task = Task.query.get_or_404(id)
    new_content = request.form['content']
    if not new_content:
        flash("Task content cannot be empty!", "error")
    else:
        task.content = new_content
        db.session.commit()
        flash("Task updated successfully!", "success")
    return redirect(url_for('index'))

@app.route('/set_priority/<int:id>', methods=['POST'])
def set_priority(id):
    task = Task.query.get_or_404(id)
    task.user_priority = int(request.form['priority'])
    db.session.commit()
    flash("Task priority updated!", "info")
    return redirect(url_for('index'))

@app.route('/delete/<int:id>', methods=['POST'])
def delete_task(id):
    task_to_delete = Task.query.get_or_404(id)
    db.session.delete(task_to_delete)
    db.session.commit()
    flash("Task deleted.", "info")
    return redirect(url_for('index'))

@app.route('/retrain-model')
def trigger_training():
    message = train_model(app, db, Task)
    flash(message, "success")
    return redirect(url_for('index'))

if __name__ == "__main__":
    if not os.path.exists('todos.db'):
        with app.app_context():
            db.create_all()
        print("Database created!")
    train_model(app, db, Task)
    app.run(debug=True)