import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;

import java.util.ArrayList;
import java.util.List;

class FoodItem {
    String name;
    double calories; // Calories par portion
    double vitaminC; // Vitamine C par portion en mg

    public FoodItem(String name, double calories, double vitaminC) {
        this.name = name;
        this.calories = calories;
        this.vitaminC = vitaminC;
    }
}

public class MealPlanner {
    public static void main(String[] args) {
        // Étape 1 : Définir vos aliments
        List<FoodItem> foodItems = new ArrayList<>();
        foodItems.add(new FoodItem("Pomme", 52, 5)); // Pomme: 52 kcal, 5 mg de vitamine C
        foodItems.add(new FoodItem("Banane", 89, 10)); // Banane: 89 kcal, 10 mg de vitamine C
        foodItems.add(new FoodItem("Poulet", 239, 0)); // Poulet: 239 kcal, 0 mg de vitamine C

        // Étape 2 : Préparer les données d'entrée
        double[][] inputData = new double[foodItems.size()][2]; // [calories, vitamine C]
        for (int i = 0; i < foodItems.size(); i++) {
            FoodItem item = foodItems.get(i);
            inputData[i][0] = item.calories; // Calories
            inputData[i][1] = item.vitaminC; // Vitamine C
        }

        // Étape 3 : Créer le modèle TensorFlow
        try (Graph graph = Graph.create()) {
            // Définir les nœuds d'entrée et de sortie
            String inputNodeName = "input"; 
            String outputNodeName = "output"; 

            // Créer des placeholders pour l'entrée et la sortie
            Tensor<Double> inputTensor = Tensor.create(Shape.of(foodItems.size(), 2), inputData);

            // Ajouter des opérations au graphe pour créer le modèle
            // Couche d'entrée : Placeholder pour les entrées
            var inputPlaceholder = graph.opBuilder("Placeholder", inputNodeName)
                    .setAttr("dtype", org.tensorflow.DataType.DOUBLE)
                    .setAttr("shape", Shape.of(-1, 2)) // Deux caractéristiques : calories et vitamine C
                    .build().output(0);

            // Couche cachée : Dense Layer (par exemple avec 64 neurones)
            var hiddenLayer = graph.opBuilder("Dense", "hidden_layer")
                    .addInput(inputPlaceholder)
                    .setAttr("units", 64)
                    .setAttr("activation", "relu")
                    .build().output(0);

            // Couche de sortie : Dense Layer pour prédire les quantités (par exemple pour chaque aliment)
            var outputLayer = graph.opBuilder("Dense", outputNodeName)
                    .addInput(hiddenLayer)
                    .setAttr("units", foodItems.size()) // Une sortie par aliment
                    .setAttr("activation", "linear") // Pas d'activation pour la sortie des quantités
                    .build().output(0);

            try (Session session = new Session(graph)) {
                // Simuler l'entraînement (ici nous ne faisons pas encore d'entraînement réel)
                for (int epoch = 0; epoch < 100; epoch++) { // Nombre d'époques à ajuster
                    session.runner()
                            .feed(inputNodeName, inputTensor)
                            .fetch(outputNodeName)
                            .run();
                }

                // Afficher les résultats après l'entraînement
                System.out.println("Quantités d'aliments recommandées :");
                Tensor<Double> outputTensor = session.runner()
                        .fetch(outputNodeName)
                        .feed(inputNodeName, inputTensor)
                        .run().get(0);

                for (int i = 0; i < foodItems.size(); i++) {
                    System.out.printf("%s: %.2f g%n", foodItems.get(i).name, outputTensor.copyTo(new double[1])[i]);
                }
            }
        }
    }
}
