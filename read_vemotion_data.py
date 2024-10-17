import os
import pandas as pd
import glob
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def read_participant_data(participant_dir):
    csv_file = glob.glob(os.path.join(participant_dir, 'export.csv'))[0]
    df = pd.read_csv(csv_file)
    return df

def read_all_participants_data(root_dir='VEmotion_data'):
    participants_data = {}
    participant_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    participant_dirs.sort()
    
    for participant in participant_dirs:
        participant_path = os.path.join(root_dir, participant)
        participants_data[participant] = read_participant_data(participant_path)
    
    return participants_data

def print_participant_info(participants_data):
    for participant, data in participants_data.items():
        print(f"Participant: {participant}")
        print(f"  Number of records: {len(data)}")
        print(f"  Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}")
        if 'expressed_emotion' in data.columns:
            print(f"  Most common expressed emotion: {data['expressed_emotion'].mode().values[0]}")
        print("  Columns:", ", ".join(data.columns))
        print()

def preprocess_data(participant_data):
    # Select relevant features (excluding emotion-related columns)
    relevant_features = [
        'Speed_gps', 'feeltemp_outside', 'windspeed', 'cloud_coverage',
        'trafficflow_reducedspeed', 'trafficflow_confidence', 'freeflow_speed',
        'max_speed', 'lanes'
    ]
    
    # Add 'road_type' if it exists in the data
    if 'road_type' in participant_data.columns:
        relevant_features.append('road_type')
    
    # Ensure all relevant features exist in the data
    existing_features = [col for col in relevant_features if col in participant_data.columns]
    
    if not existing_features:
        raise ValueError("No relevant features found in the data.")
    
    X = participant_data[existing_features]
    
    # Handle categorical variables
    if 'road_type' in X.columns:
        X = pd.get_dummies(X, columns=['road_type'])
    
    # Ensure 'expressed_emotion' column exists
    if 'expressed_emotion' not in participant_data.columns:
        raise ValueError("'expressed_emotion' column not found in the data.")
    
    y = participant_data['expressed_emotion']
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le, X.columns.tolist()

def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")
    
    return clf

def extract_rules(clf, feature_names, class_names):
    tree = clf.tree_
    rules = []

    def recurse(node, depth, parent_rule):
        if tree.feature[node] != -2:  # not a leaf node
            name = feature_names[tree.feature[node]].lower()
            threshold = tree.threshold[node]
            
            left_rule = f"{parent_rule}IF {name} <= {threshold:.2f} THEN "
            recurse(tree.children_left[node], depth + 1, left_rule)
            
            right_rule = f"{parent_rule}IF {name} > {threshold:.2f} THEN "
            recurse(tree.children_right[node], depth + 1, right_rule)
        else:
            class_index = tree.value[node].argmax()
            rules.append(f"{parent_rule}expressed_emotion = {class_names[class_index]}")

    recurse(0, 1, "")
    return rules

def print_rules(rules):
    print("Generated IF...THEN Rules:")
    for i, rule in enumerate(rules, 1):
        print(f"{i}. {rule}")

def apply_rules(rules, data):
    for rule in rules:
        parts = rule.split(" THEN ")
        conditions = parts[0]
        result = parts[-1]  # Take the last part as the result
        
        conditions = conditions.split(" AND ")
        
        if all(eval(cond.replace("IF ", ""), {"__builtins__": None}, data) for cond in conditions):
            return result.split(" = ")[1]
    
    return "Unknown"  # Default case if no rule matches

def train_and_extract_rules_for_participant(participant_data, participant_id):
    X, y_encoded, le, feature_names = preprocess_data(participant_data)
    
    if len(np.unique(y_encoded)) < 2:
        print(f"Participant {participant_id} has only one emotion class. Skipping rule extraction.")
        return None
    
    clf = train_decision_tree(X, y_encoded)
    
    class_names = le.classes_
    
    rules = extract_rules(clf, feature_names, class_names)
    return rules

if __name__ == "__main__":
    all_data = read_all_participants_data()
    print_participant_info(all_data)
    
    last_successful_features = None
    
    for participant_id, participant_data in all_data.items():
        print(f"\nProcessing Participant: {participant_id}")
        try:
            X, y_encoded, le, feature_names = preprocess_data(participant_data)
            rules = train_and_extract_rules_for_participant(participant_data, participant_id)
            
            if rules:
                print(f"Rules for Participant {participant_id}:")
                print_rules(rules)
                last_successful_features = feature_names
            
            print("-" * 50)
        except Exception as e:
            print(f"Error processing participant {participant_id}: {str(e)}")
            print("-" * 50)
    
    # Test the rules on a sample data point (using the last successful participant's features)
    if last_successful_features:
        sample_data = {col: 0 for col in last_successful_features}
        sample_data['Speed_gps'] = 50
        sample_data['feeltemp_outside'] = 20
        sample_data['windspeed'] = 10
        sample_data['cloud_coverage'] = 50
        sample_data['trafficflow_reducedspeed'] = 0.8
        sample_data['trafficflow_confidence'] = 0.9
        sample_data['freeflow_speed'] = 60
        sample_data['max_speed'] = 70
        sample_data['lanes'] = 2
        if 'road_type_highway' in sample_data:
            sample_data['road_type_highway'] = 1
        
        result = apply_rules(rules, sample_data)
        print(f"\nPredicted expressed emotion for sample data (using rules from Participant {participant_id}): {result}")
    else:
        print("\nNo successful rule extraction. Unable to test on sample data.")
