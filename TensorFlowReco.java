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

        // Étape 3 : Définir les objectifs
        double targetCalories = 2000; // Objectif calorique
        double targetVitaminC = 2000; // Objectif en vitamine C

        // Étape 4 : Créer le modèle TensorFlow
        try (Graph graph = Graph.create()) {
            String inputNodeName = "input"; 
            String outputNodeName = "output"; 

            Tensor<Double> inputTensor = Tensor.create(Shape.of(foodItems.size(), 2), inputData);

            // Créer des placeholders pour l'entrée et la sortie
            //Un placeholder est un type de nœud dans TensorFlow qui permet de définir une variable d'entrée pour votre modèle.
            //Il ne contient pas de données lui-même, mais il attend que des données soient fournies lors de l'exécution du modèle.
            var inputPlaceholder = graph.opBuilder("Placeholder", inputNodeName)
                    .setAttr("dtype", org.tensorflow.DataType.DOUBLE) //Type de valeurs en entrées (ici des mg ou kcal, on utlise le type double)
                    .setAttr("shape", Shape.of(-1, 2)) // le placeholder peut accepter un nombre variable d'exemples (d'où le -1) et deux caractéristiques par exemple (calories et vitamine C)
                    .build().output(0);

            // Couche cachée : Dense Layer avec activation ReLU
            var hiddenLayer = graph.opBuilder("Dense", "hidden_layer")
                    .addInput(inputPlaceholder) //Creation de la couche avec le paceholder définit plus haut
                    .setAttr("units", 64) //nombre de neurones dans la couche dense
                    .setAttr("activation", "relu") // fonction d'activation ReLU (Rectified Linear Unit), qui est couramment utilisée dans les réseaux de neurones car elle aide à introduire de la non-linéarité dans le modèle tout en étant efficace en termes de calcul.
                    .build().output(0);

            // Couche de sortie : Dense Layer pour prédire les quantités
            var outputLayer = graph.opBuilder("Dense", outputNodeName)
                    .addInput(hiddenLayer)
                    .setAttr("units", foodItems.size())  //Nombre de neurone en sortie (ie nbr d'aliments proposés). Faut pas en prendre trop donc
                    .setAttr("activation", "linear") //approprié lorsque vous souhaitez prédire des valeurs continues (comme des quantités en grammes)
                    .build().output(0);

            try (Session session = new Session(graph)) {
                // Simuler l'entraînement (ici nous ne faisons pas encore d'entraînement réel)
                for (int epoch = 0; epoch < 100; epoch++) { 
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

                // Vérifiez si les résultats atteignent les objectifs
                double totalCalories = calculateTotalCalories(outputTensor);
                double totalVitaminC = calculateTotalVitaminC(outputTensor, foodItems);
                
                System.out.printf("Total Calories: %.2f g%n", totalCalories);
                System.out.printf("Total Vitamin C: %.2f g%n", totalVitaminC);
                
                if(totalCalories >= targetCalories) {
                    System.out.println("Objectif calorique atteint !");
                } else {
                    System.out.println("Objectif calorique non atteint.");
                }
                
                if(totalVitaminC >= targetVitaminC) {
                    System.out.println("Objectif en vitamine C atteint !");
                } else {
                    System.out.println("Objectif en vitamine C non atteint.");
                }
            }
        }
    }

    private static double calculateTotalCalories(Tensor<Double> outputTensor) {
        double totalCalories = 0.0;
        for(int i=0; i<outputTensor.shape(0); i++) {
            totalCalories += outputTensor.copyTo(new double[1])[i]; // Remplacez par la logique pour calculer les calories totales
        }
        return totalCalories;
    }

    private static double calculateTotalVitaminC(Tensor<Double> outputTensor, List<FoodItem> foodItems) {
        double totalVitaminC = 0.0;
        for(int i=0; i<outputTensor.shape(0); i++) {
            totalVitaminC += outputTensor.copyTo(new double[1])[i] * foodItems.get(i).vitaminC / 100; // Ajustez selon la quantité en grammes
        }
        return totalVitaminC;
    }
}
