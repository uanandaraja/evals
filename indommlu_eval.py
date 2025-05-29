import json
import os
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_mcq(item, model, client):
    """Evaluate a single multiple choice question"""
    prompt = f"""Ini adalah soal {item['subject']} untuk {item['level']}. Pilihlah salah satu jawaban yang dianggap benar!

{item['soal']}
{item['jawaban']}

Jawab HANYA dengan huruf pilihan saja (A, B, C, D, atau E). Jangan tambahkan penjelasan awal atau akhir. Hanya output huruf pilihan saja.
Jawaban:"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )
    
    predicted = response.choices[0].message.content.strip()
    correct = item['kunci']
    
    return {
        'id': item['id'],
        'predicted': predicted,
        'correct': correct,
        'is_correct': predicted == correct
    }

def main():
    # Configure OpenAI client for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key")
        return
    
    # Models to evaluate
    models = [
        "anthropic/claude-sonnet-4"
    ]
    
    # Load and filter data
    print("Loading data...")
    try:
        data = load_jsonl('indoMMLU.jsonl')
    except FileNotFoundError:
        print("Error: indoMMLU.jsonl file not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    filtered_data = [item for item in data if item['level'] == 'Seleksi PTN' and item['is_for_fewshot'] == '0']
    print(f"Loaded {len(filtered_data)} questions for evaluation")
    
    all_model_results = {}
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model}")
        print(f"{'='*50}")
        
        results = []
        correct_count = 0
        
        for i, item in enumerate(tqdm(filtered_data, desc=f"Evaluating {model}")):
            try:
                result = evaluate_mcq(item, model, client)
                
                # Add more details to result
                result.update({
                    'model': model,
                    'subject': item['subject'],
                    'soal': item['soal'],
                    'jawaban': item['jawaban'],
                    'sumber': item['sumber']
                })
                
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                
                current_accuracy = correct_count / len(results)
                
                # Show first 10 outputs
                if i < 10:
                    status = "✓" if result['is_correct'] else "✗"
                    print(f"\nQuestion {i+1} ({item['subject']}):")
                    print(f"Predicted: {result['predicted']} | Correct: {result['correct']} {status}")
                    print(f"Question: {item['soal'][:100]}...")
                    print(f"Options: {item['jawaban']}")
                    print(f"Running accuracy: {current_accuracy:.3f}")
                    print("-" * 60)
                
                # Show progress every 50 questions
                if (i + 1) % 50 == 0:
                    print(f"\nProgress: {i+1}/{len(filtered_data)} | Accuracy: {current_accuracy:.3f}")
                    
            except Exception as e:
                print(f"Error evaluating question {i+1}: {e}")
                continue
        
        # Save results for this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.replace('/', '_')
        output_file = f"eval_results_{model_name}_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Store results
        final_accuracy = sum(r['is_correct'] for r in results) / len(results) if results else 0
        all_model_results[model] = {
            'accuracy': final_accuracy,
            'results': results,
            'output_file': output_file
        }
        
        print(f"Final accuracy for {model}: {final_accuracy:.3f}")
        print(f"Results saved to: {output_file}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    for model, data in all_model_results.items():
        print(f"{model}: {data['accuracy']:.3f}")

if __name__ == "__main__":
    main() 
