export interface Prediction {
  class_name: string;
  confidence: number;
}

export interface StreamResult {
  predictions: Prediction[];
  timestamp: number;
  audio_duration: number;
}

export interface HistoryEntry {
  epoch: number;
  tr_loss: number;
  tr_acc: number;
  val_loss: number;
  val_acc: number;
  lr?: number;
}

export interface ModelInfo {
  model_name: string;
  num_classes: number;
  total_params: number;
  best_epoch: number;
  best_val_acc: number;
  total_epochs: number;
  history: HistoryEntry[];
  class_names: string[];
}

export interface ClassAccuracy {
  idx: number;
  name: string;
  accuracy: number;
  correct: number;
  total: number;
}

export interface EvaluationResult {
  overall_accuracy: number;
  per_class: ClassAccuracy[];
}
