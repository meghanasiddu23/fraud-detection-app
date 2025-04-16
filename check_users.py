from app import app, db, User  # Import app, db, and User model

# Push the application context before querying
with app.app_context():
    users = User.query.all()
    
    if users:
        for user in users:
            print(f"Username: {user.username}, Hashed Password: {user.password}")
    else:
        print("No users found in the database.")
