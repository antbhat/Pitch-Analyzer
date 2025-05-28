import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
import google.generativeai as genai
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split # Optional: if you want to test model performance

# --- Configuration ---
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY = ""  # <--- REPLACE WITH YOUR GEMINI API KEY

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Please set it as an environment variable or directly in the script.")
genai.configure(api_key=GOOGLE_API_KEY)

PITCH_DECKS_DIR = Path("/Users/akshubhat/Downloads/All_PAN_Data/Training Data/Pitch Deck") # Used for finding training PDFs and new PDFs to score

# --- Ground Truth Data (Transcribed from your image) ---
# IMPORTANT: You MUST complete this JSON with all data from your image accurately.
# Company names here should ideally match PDF filenames in PITCH_DECKS_DIR (e.g., "Statista.pdf")
GROUND_TRUTH_JSON_STRING = """
[
  {
    "company_name": "Valiant",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 5,
        "Purpose Integration into Decision Making": 5,
        "Purpose Communication": 5
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 4,
        "Innovation": 3,
        "Speed of Iteration": 4
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 2,
        "Profitable": 0,
        "Customer Traction": 1,
        "Gross Margin": 2,
        "Velocity": 3
      }
    }
  },
  {
    "company_name": "Puka Up",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 5,
        "Purpose Integration into Decision Making": 5,
        "Purpose Communication": 5
      },
      "Transformation": {
        "Transformation Potential": 4,
        "People Empowered": 2,
        "Innovation": 2,
        "Speed of Iteration": 1
      },
      "Performance": {
        "How much transformational potentail has been realised": 2,
        "Revenue": 2,
        "Profitable": 1,
        "Customer Traction": 3,
        "Gross Margin": 3,
        "Velocity": 3
      }
    }
  },
  {
    "company_name": "Care Guidance",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 5,
        "Purpose Integration into Decision Making": 4,
        "Purpose Communication": 5
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 2,
        "Innovation": 1,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 2,
        "Revenue": 2,
        "Profitable": 2,
        "Customer Traction": 3,
        "Gross Margin": 3,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "WeedOUT",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 5,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 4
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 5,
        "Innovation": 5,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 0,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 0,
        "Gross Margin": 0,
        "Velocity": 0
      }
    }
  },
  {
    "company_name": "Quasar",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 1,
        "Purpose Integration into Decision Making": 1,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 5,
        "Innovation": 5,
        "Speed of Iteration": 1
      },
      "Performance": {
        "How much transformational potentail has been realised": 0,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 0,
        "Gross Margin": 0,
        "Velocity": 0
      }
    }
  },
  {
    "company_name": "Ferronova",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 3
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 3,
        "Innovation": 5,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 3,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 2,
        "Gross Margin": 0,
        "Velocity": 1
      }
    }
  },
  {
    "company_name": "Kinoxis",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 4,
        "Purpose Communication": 4
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 5,
        "Innovation": 5,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 3,
        "Revenue": 1,
        "Profitable": 0,
        "Customer Traction": 2,
        "Gross Margin": 0,
        "Velocity": 1
      }
    }
  },
  {
    "company_name": "The Ripple Group",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 5,
        "Purpose Integration into Decision Making": 2,
        "Purpose Communication": 5
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 4,
        "Innovation": 3,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 3,
        "Revenue": 4,
        "Profitable": 3,
        "Customer Traction": 3,
        "Gross Margin": 3,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Phyllome",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 2,
        "Purpose Communication": 3
      },
      "Transformation": {
        "Transformation Potential": 4,
        "People Empowered": 4,
        "Innovation": 4,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 2,
        "Revenue": 1,
        "Profitable": 0,
        "Customer Traction": 3,
        "Gross Margin": 1,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "AMBIT",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 1,
        "Purpose Integration into Decision Making": 0,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 2,
        "People Empowered": 3,
        "Innovation": 2,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 1,
        "Gross Margin": 0,
        "Velocity": 0
      }
    }
  },
  {
    "company_name": "BugBiome",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 4,
        "Purpose Integration into Decision Making": 4,
        "Purpose Communication": 4
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 4,
        "Innovation": 4,
        "Speed of Iteration": 4
      },
      "Performance": {
        "How much transformational potentail has been realised": 0,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 0,
        "Gross Margin": 0,
        "Velocity": 0
      }
    }
  },
  {
    "company_name": "Goterra",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 1,
        "Purpose Integration into Decision Making": 0,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 4,
        "Innovation": 2,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 2,
        "Revenue": 2,
        "Profitable": 0,
        "Customer Traction": 2,
        "Gross Margin": 1,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Uluu",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 4,
        "Purpose Communication": 2
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 5,
        "Innovation": 5,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 1,
        "Profitable": 0,
        "Customer Traction": 2,
        "Gross Margin": 1,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Flux",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 4,
        "Purpose Integration into Decision Making": 4,
        "Purpose Communication": 4
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 5,
        "Innovation": 4,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 0,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 1,
        "Gross Margin": 1,
        "Velocity": 1
      }
    }
  },
  {
    "company_name": "PAN",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 5,
        "Purpose Integration into Decision Making": 5,
        "Purpose Communication": 5
      },
      "Transformation": {
        "Transformation Potential": 5,
        "People Empowered": 5,
        "Innovation": 5,
        "Speed of Iteration": 5
      },
      "Performance": {
        "How much transformational potentail has been realised": 2,
        "Revenue": 3,
        "Profitable": 3,
        "Customer Traction": 5,
        "Gross Margin": 2,
        "Velocity": 5
      }
    }
  },
  {
    "company_name": "Gridsoft",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 1,
        "Purpose Integration into Decision Making": 1,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 1,
        "Innovation": 2,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 1,
        "Gross Margin": 1,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Hydgene",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 2
      },
      "Transformation": {
        "Transformation Potential": 4,
        "People Empowered": 3,
        "Innovation": 5,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 2,
        "Gross Margin": 2,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Flipped Energy",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 1,
        "Purpose Integration into Decision Making": 1,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 3,
        "Innovation": 3,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 0,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 0,
        "Gross Margin": 0,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "1s1 Energy",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 4
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 3,
        "Innovation": 5,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 1,
        "Profitable": 0,
        "Customer Traction": 1,
        "Gross Margin": 0,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Real View Imaging",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 2,
        "Purpose Integration into Decision Making": 2,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 4,
        "Innovation": 3,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 2,
        "Revenue": 1,
        "Profitable": 0,
        "Customer Traction": 2,
        "Gross Margin": 0,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Messium",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 4,
        "Purpose Integration into Decision Making": 2,
        "Purpose Communication": 3,
        "Highest Score": 15
      },
      "Transformation": {
        "Transformation Potential": 4,
        "People Empowered": 4,
        "Innovation": 2,
        "Speed of Iteration": 3,
        "Highest Score": 20
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 0,
        "Profitable": 2,
        "Customer Traction": 3,
        "Gross Margin": 3,
        "Velocity": 3
      }
    }
  },
  {
    "company_name": "Macrobiome",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 4,
        "People Empowered": 4,
        "Innovation": 4,
        "Speed of Iteration": 1
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 0,
        "Gross Margin": 0,
        "Velocity": 0
      }
    }
  },
  {
    "company_name": "Akson Robotics",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 3,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 2
      },
      "Transformation": {
        "Transformation Potential": 3,
        "People Empowered": 3,
        "Innovation": 3,
        "Speed of Iteration": 3
      },
      "Performance": {
        "How much transformational potentail has been realised": 3,
        "Revenue": 1,
        "Profitable": 0,
        "Customer Traction": 3,
        "Gross Margin": 3,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Jellagen",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 1,
        "Purpose Integration into Decision Making": 1,
        "Purpose Communication": 1
      },
      "Transformation": {
        "Transformation Potential": 4,
        "People Empowered": 4,
        "Innovation": 3,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 0,
        "Profitable": 0,
        "Customer Traction": 0,
        "Gross Margin": 0,
        "Velocity": 2
      }
    }
  },
  {
    "company_name": "Sokerdata",
    "scores": {
      "Purpose": {
        "Purpose Consistency": 4,
        "Purpose Integration into Decision Making": 3,
        "Purpose Communication": 3
      },
      "Transformation": {
        "Transformation Potential": 2,
        "People Empowered": 1,
        "Innovation": 2,
        "Speed of Iteration": 2
      },
      "Performance": {
        "How much transformational potentail has been realised": 1,
        "Revenue": 2,
        "Profitable": 1,
        "Customer Traction": 2,
        "Gross Margin": 2,
        "Velocity": 3
      }
    }
  }
]
"""

# --- Define Scoring Criteria (Order matters for flattening/unflattening scores) ---
# This order must be consistent.
ORDERED_CRITERIA_FLAT = [
    "Purpose Consistency", "Purpose Integration into Decision Making", "Purpose Communication",
    "Transformation Potential", "People Empowered", "Innovation", "Speed of Iteration",
    "How much transformational potential has been realized", "Revenue", "Profitable",
    "Customer Traction", "Gross Margin", "Velocity"
]

# Mapping sections to their criteria for structured output
STRUCTURED_CRITERIA_MAP = {
    "Purpose": ORDERED_CRITERIA_FLAT[0:3],
    "Transformation": ORDERED_CRITERIA_FLAT[3:7],
    "Performance": ORDERED_CRITERIA_FLAT[7:13]
}

# --- Initialize Gemini Model (primarily for embeddings) ---
try:
    # Using a model good for embeddings, or a specific embedding model
    embedding_model_name = "models/text-embedding-004" # Dedicated embedding model
    print(f"Using embedding model: {embedding_model_name}")
except Exception as e:
    print(f"Error related to embedding model name (this part is just for reference): {e}")
    # embedding_model_name will be used directly in genai.embed_content

# --- Helper Functions ---
def load_pitch_deck_text(pdf_path: Path) -> str | None:
    """Loads all text from a PDF file."""
    if not pdf_path.exists():
        print(f"  PDF not found: {pdf_path}")
        return None
    try:
        print(f"  Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        full_text = " ".join([page.page_content for page in pages]).strip()
        return full_text
    except Exception as e:
        print(f"  Error loading or reading PDF {pdf_path.name}: {e}")
        return None

def generate_text_embedding(text: str) -> List[float] | None:
    """Generates embedding for a given text using Gemini."""
    if not text:
        return None
    try:
        # print(f"    Generating embedding for text snippet (len: {len(text)})...")
        result = genai.embed_content(
            model=embedding_model_name, # Use the defined embedding model
            content=text,
            task_type="RETRIEVAL_DOCUMENT" # Or "SEMANTIC_SIMILARITY"
        )
        time.sleep(0.2) # Small delay to respect embedding API rate limits
        return result['embedding']
    except Exception as e:
        print(f"    Error generating embedding: {e}")
        return None

def flatten_scores(structured_scores: Dict[str, Dict[str, int]]) -> List[int | float] | None:
    """Flattens structured scores into a list based on ORDERED_CRITERIA_FLAT."""
    flat_scores = []
    try:
        for criterion_name in ORDERED_CRITERIA_FLAT:
            found = False
            for section_data in structured_scores.values():
                if criterion_name in section_data:
                    flat_scores.append(section_data[criterion_name])
                    found = True
                    break
            if not found:
                print(f"Warning: Criterion '{criterion_name}' not found in structured scores during flattening.")
                flat_scores.append(0) # Or handle as error
        return flat_scores
    except Exception as e:
        print(f"Error flattening scores: {e}")
        return None


def unflatten_scores_to_structured_dict(flat_scores: List[float]) -> Dict[str, Any]:
    """Converts a flat list of predicted scores back to the structured format."""
    structured = {}
    current_idx = 0
    for section, criteria_in_section in STRUCTURED_CRITERIA_MAP.items():
        structured[section] = {}
        section_numeric_scores = []
        for criterion_name in criteria_in_section:
            if current_idx < len(flat_scores):
                # Predicted scores might be float, round them or use as is
                pred_score = round(flat_scores[current_idx]) # Example: rounding
                # Clamp scores to a valid range if necessary, e.g., 0-5 or 0-10
                pred_score = max(0, min(10, pred_score)) # Assuming 0-10 scale from example
                structured[section][criterion_name] = {
                    "score": pred_score,
                    "justification": "Predicted by supervised model" # Supervised model doesn't give text justification
                }
                if isinstance(pred_score, (int, float)):
                     section_numeric_scores.append(pred_score)
            else: # Should not happen if ORDERED_CRITERIA_FLAT is correct
                 structured[section][criterion_name] = {"score": 0, "justification": "Error: Missing prediction"}
            current_idx += 1
        
        if section_numeric_scores:
            structured[section]["Highest Score"] = round(max(section_numeric_scores))
        else:
            structured[section]["Highest Score"] = 0
            
    return structured


# --- Main Processing Logic ---
def main():
    # 1. Load and Parse Ground Truth Data
    try:
        ground_truth_data = json.loads(GROUND_TRUTH_JSON_STRING)
        if not isinstance(ground_truth_data, list) or not all(isinstance(item, dict) for item in ground_truth_data):
            raise ValueError("GROUND_TRUTH_JSON_STRING is not a list of dictionaries.")
        print(f"Successfully loaded {len(ground_truth_data)} entries from ground truth JSON.")
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Could not parse GROUND_TRUTH_JSON_STRING: {e}")
        print("Please ensure the JSON string is valid.")
        return
    except ValueError as e:
        print(f"CRITICAL ERROR: Invalid structure in GROUND_TRUTH_JSON_STRING: {e}")
        return

    # 2. Prepare Training Data for Supervised Model
    X_train_embeddings: List[List[float]] = []
    y_train_flat_scores: List[List[int | float]] = []
    processed_training_companies = []

    print("\n--- Preparing Training Data ---")
    for entry in ground_truth_data:
        company_name = entry.get("company_name")
        structured_scores = entry.get("scores")

        if not company_name or not structured_scores:
            print(f"Skipping entry due to missing company_name or scores: {entry}")
            continue

        # Try to find matching PDF (simple name matching)
        # You might need more sophisticated matching logic
        pdf_path = PITCH_DECKS_DIR / f"{company_name}.pdf"
        pitch_text = load_pitch_deck_text(pdf_path)

        if pitch_text:
            embedding = generate_text_embedding(pitch_text)
            flat_scores = flatten_scores(structured_scores)

            if embedding and flat_scores and len(flat_scores) == len(ORDERED_CRITERIA_FLAT):
                X_train_embeddings.append(embedding)
                y_train_flat_scores.append(flat_scores)
                processed_training_companies.append(company_name)
                print(f"  Added '{company_name}' to training set.")
            else:
                print(f"  Skipping '{company_name}': Failed to get embedding or flatten scores correctly.")
        else:
            print(f"  Skipping '{company_name}': PDF not found or could not be read ({pdf_path}).")

    if not X_train_embeddings or not y_train_flat_scores:
        print("\nCRITICAL ERROR: No training data could be prepared. Reasons could be:")
        print("- GROUND_TRUTH_JSON_STRING is empty or malformed.")
        print("- No matching PDFs found in PITCH_DECKS_DIR for companies in JSON.")
        print("- Errors during text embedding or score flattening.")
        print("Supervised model cannot be trained. Exiting.")
        return

    X_train = np.array(X_train_embeddings)
    y_train = np.array(y_train_flat_scores)

    print(f"\nPrepared training data with {X_train.shape[0]} samples.")
    print(f"Feature vector dimension: {X_train.shape[1]}")
    print(f"Target vector dimension: {y_train.shape[1]}")


    # 3. Train Supervised Model
    print("\n--- Training Supervised Model ---")
    try:
        # Using RandomForestRegressor as it's robust and handles non-linearities
        # n_estimators: number of trees, random_state for reproducibility
        # n_jobs=-1 uses all available cores
        base_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)
        supervised_model = MultiOutputRegressor(base_regressor)
        
        # Optional: Split training data to evaluate model if you have enough samples
        # X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        # supervised_model.fit(X_fit, y_fit)
        # score = supervised_model.score(X_eval, y_eval) # R^2 score
        # print(f"Model evaluation R^2 score on hold-out set: {score_eval:.3f}")
        
        supervised_model.fit(X_train, y_train)
        print("Supervised model trained successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to train supervised model: {e}")
        return

    # 4. Predict Scores for Pitch Decks using the Trained Supervised Model
    all_decks_predicted_results: List[Dict[str, Any]] = []
    print(f"\n--- Predicting Scores for Decks in {PITCH_DECKS_DIR} using Supervised Model ---")

    pitch_deck_files_to_score = list(PITCH_DECKS_DIR.glob("*.pdf"))
    if not pitch_deck_files_to_score:
        print(f"No PDF files found in {PITCH_DECKS_DIR} to score with the supervised model.")
        return

    for i, pdf_path in enumerate(pitch_deck_files_to_score):
        print(f"\nProcessing for Prediction ({i+1}/{len(pitch_deck_files_to_score)}): {pdf_path.name}")
        
        # Check if this deck was part of the explicit training set, can optionally skip or just predict
        # company_name_from_pdf = pdf_path.stem # Gets filename without extension
        # if company_name_from_pdf in processed_training_companies:
        #     print(f"  Note: '{company_name_from_pdf}' was in the training set. Predicting anyway for demonstration.")

        pitch_text = load_pitch_deck_text(pdf_path)
        if not pitch_text:
            all_decks_predicted_results.append({"File Name": pdf_path.name, "Prediction Status": "Error loading PDF"})
            continue

        embedding = generate_text_embedding(pitch_text)
        if not embedding:
            all_decks_predicted_results.append({"File Name": pdf_path.name, "Prediction Status": "Error generating embedding"})
            continue

        # Predict scores using the trained supervised model
        predicted_flat_scores = supervised_model.predict(np.array([embedding]))[0] # predict expects 2D array

        # Unflatten scores and prepare for CSV
        predicted_structured_evaluations = unflatten_scores_to_structured_dict(predicted_flat_scores)
        
        current_deck_output: Dict[str, Any] = {"File Name": pdf_path.name, "Prediction Status": "Success"}
        for section, criteria_data in predicted_structured_evaluations.items():
            for crit_name, eval_data in criteria_data.items():
                if crit_name == "Highest Score":
                     current_deck_output[f"{section} Highest Score"] = eval_data
                else:
                    current_deck_output[f"{crit_name} Score"] = eval_data.get("score", 0)
                    current_deck_output[f"{crit_name} Explanation"] = eval_data.get("justification", "")
        
        all_decks_predicted_results.append(current_deck_output)
        print(f"  Successfully predicted scores for {pdf_path.name}")
        # No per-deck API call for scoring, so sleep is mainly for embedding calls if many new decks
        # time.sleep(0.5) # Small pause if processing many new decks rapidly

    # 5. Output Results to CSV
    if all_decks_predicted_results:
        try:
            df = pd.DataFrame(all_decks_predicted_results)
            
            # Define column order for better readability
            ordered_columns = ["File Name", "Prediction Status"]
            for section, criteria_list_names in STRUCTURED_CRITERIA_MAP.items():
                for criterion_name_key in criteria_list_names: # These are keys from ORDERED_CRITERIA_FLAT
                    ordered_columns.append(f"{criterion_name_key} Score")
                    ordered_columns.append(f"{criterion_name_key} Explanation")
                ordered_columns.append(f"{section} Highest Score")
            
            # Reindex to ensure all columns are present and in order, fill missing with N/A
            df = df.reindex(columns=ordered_columns, fill_value="N/A")
            
            output_csv_path = "empowerment_scores_supervised_model.csv"
            df.to_csv(output_csv_path, index=False)
            print(f"\n✅ Supervised model predictions saved to {output_csv_path}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")
            print("Dumping results to console instead:")
            for res_item in all_decks_predicted_results:
                print(res_item)
    else:
        print("\nNo supervised predictions generated to save.")

if __name__ == "__main__":
    main()
