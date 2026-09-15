"use client";

import React, { useState } from "react";
import { Navbar, TabKey } from "@/components/Navbar";
import { MetricCard } from "@/components/MetricCard";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";
import { TrainingCurves } from "@/components/TrainingCurves";
import { FeatureDistributions } from "@/components/FeatureDistributions";
import { Predictor } from "@/components/Predictor";
import { FnnModelArtifact } from "@/lib/inference";

// Statically import exported production artifacts
import rawModel from "@/public/model/model.json";
import rawDataset from "@/public/data/dataset.json";
import rawPrep from "@/public/data/preprocessing.json";
import rawTrain from "@/public/data/training_history.json";
import rawEval from "@/public/data/evaluation.json";

import {
  Brain,
  Database,
  Layers,
  Activity,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Search,
  ChevronRight,
  Sparkles,
} from "lucide-react";

const modelData = rawModel as unknown as FnnModelArtifact;
const datasetData = rawDataset as Array<{
  id: number;
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
  target: number;
  species: string;
}>;
const prepData = rawPrep;
const trainData = rawTrain;
const evalData = rawEval;

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("home");

  // Dataset Table State
  const [searchTerm, setSearchTerm] = useState("");
  const [speciesFilter, setSpeciesFilter] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Filtered dataset
  const filteredData = datasetData.filter((item) => {
    const matchesSpecies = speciesFilter === "all" || item.species.toLowerCase() === speciesFilter.toLowerCase();
    const matchesSearch =
      searchTerm === "" ||
      item.id.toString().includes(searchTerm) ||
      item.species.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSpecies && matchesSearch;
  });

  const totalPages = Math.ceil(filteredData.length / pageSize);
  const paginatedData = filteredData.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div>
      <Navbar activeTab={activeTab} onSelectTab={setActiveTab} />

      <main>
        {/* ==================================================================== */}
        {/* TAB 1: HOME */}
        {/* ==================================================================== */}
        {activeTab === "home" && (
          <section className="space-y-8">
            <div className="rounded-2xl border border-slate-200 bg-white p-7 shadow-sm sm:p-8">
              <div className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
                <Sparkles className="h-3.5 w-3.5" />
                Academic Pattern Recognition Dashboard
              </div>
              <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl">
                Deep Feedforward Neural Network for Classification
              </h2>
              <p className="mt-1 text-xs font-bold uppercase tracking-wider text-slate-500">
                Course: Pattern Recognition &bull; Assessment: TAE 1: Project Based Learning &bull; Phase I
              </p>

              <div className="mt-4 text-base font-semibold text-blue-600">
                From input features to intelligent multi-class pattern classification
              </div>
              <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-600">
                A Deep Feedforward Neural Network (FNN) processes data strictly in one direction through fully connected
                (Dense) layers. Built with TensorFlow/Keras and deployed as a high-performance Next.js application on Vercel,
                this project demonstrates complete pattern classification on the classic Iris flower benchmark dataset.
              </p>

              {/* 3 Metric Cards */}
              <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
                <MetricCard title="Dataset" value="Iris Flower" subtitle="150 Balanced Samples" />
                <MetricCard title="Model" value="Deep FNN" subtitle="515 Trainable Parameters" highlight />
                <MetricCard title="Algorithm" value="Adam + ReLU" subtitle="Softmax Multi-Class Output" />
              </div>
            </div>

            {/* Machine Learning Pipeline Flowchart */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                End-to-End Machine Learning Pipeline
              </h3>

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-6">
                {[
                  { step: "1. Dataset", desc: "150 Iris Samples" },
                  { step: "2. Preprocessing", desc: "StandardScaler" },
                  { step: "3. Architecture", desc: "4-Layer FNN" },
                  { step: "4. Training", desc: "Mini-Batch Adam" },
                  { step: "5. Evaluation", desc: "Holdout Testing" },
                  { step: "6. Prediction", desc: "Real-Time Inference", highlight: true },
                ].map((item, idx) => (
                  <div
                    key={item.step}
                    className={`flex flex-col items-center justify-center rounded-xl border p-3.5 text-center ${
                      item.highlight
                        ? "border-blue-300 bg-blue-50/80 text-blue-700 shadow-sm"
                        : "border-slate-200 bg-slate-50 text-slate-700"
                    }`}
                  >
                    <div className="text-xs font-bold">{item.step}</div>
                    <div className="mt-0.5 text-[11px] text-slate-500">{item.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Technical Foundations Card */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-sm font-bold text-slate-900">Key Neural Network Concepts</h3>
                <ul className="mt-3 space-y-2.5 text-xs text-slate-600">
                  <li className="flex items-start gap-2">
                    <span className="font-bold text-blue-600">&bull;</span>
                    <span>
                      <b>Unidirectional Feedforward Flow:</b> Signals propagate strictly forward from input to output (x &rarr; h<sub>1</sub> &rarr; h<sub>2</sub> &rarr; h<sub>3</sub> &rarr; y&#770;) without cyclical or feedback connections.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-bold text-blue-600">&bull;</span>
                    <span>
                      <b>ReLU Hidden Activations:</b> a = max(0, z) avoids gradient saturation and enables computationally efficient piecewise-linear representation.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-bold text-blue-600">&bull;</span>
                    <span>
                      <b>Softmax Probabilities:</b> Normalizes final logits into mutually exclusive probabilities strictly summing to 1.0.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-bold text-blue-600">&bull;</span>
                    <span>
                      <b>Categorical Cross-Entropy:</b> Loss objective that yields linear backpropagation error gradients (y&#770; - y).
                    </span>
                  </li>
                </ul>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-sm font-bold text-slate-900">Student & Technology Stack</h3>
                <div className="mt-3 space-y-2 text-xs text-slate-600">
                  <div className="flex justify-between border-b border-slate-100 py-1.5">
                    <span className="text-slate-500">Student Name:</span>
                    <span className="font-semibold text-slate-900">Amogh Samarth</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-100 py-1.5">
                    <span className="text-slate-500">USN:</span>
                    <span className="font-semibold font-mono text-slate-900">CM23034</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-100 py-1.5">
                    <span className="text-slate-500">Framework:</span>
                    <span className="font-semibold text-slate-900">Next.js 15 &bull; React 19 &bull; TypeScript</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-100 py-1.5">
                    <span className="text-slate-500">Training Backend:</span>
                    <span className="font-semibold text-slate-900">TensorFlow 2.x &bull; Keras &bull; Scikit-learn</span>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <span className="text-slate-500">Deployment Target:</span>
                    <span className="font-semibold text-blue-600">Vercel (Zero Python Serverless Overhead)</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ==================================================================== */}
        {/* TAB 2: DATASET */}
        {/* ==================================================================== */}
        {activeTab === "dataset" && (
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Iris Flower Dataset
              </h2>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Benchmark Multi-Class Pattern Recognition Data (Fisher, 1936)
              </p>
            </div>

            {/* 4 Metric Cards */}
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <MetricCard title="Total Samples" value="150" subtitle="50 per class (Balanced)" />
              <MetricCard title="Features" value="4" subtitle="Continuous (cm)" />
              <MetricCard title="Classes" value="3" subtitle="Setosa, Versicolor, Virginica" highlight />
              <MetricCard title="Missing Values" value="0" subtitle="100% Complete Data" />
            </div>

            {/* Feature Distribution Visualizer */}
            <div className="space-y-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Feature Distributions by Flower Species
              </h3>
              <FeatureDistributions dataset={datasetData} />
            </div>

            {/* Interactive Data Table */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-4">
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
                    Dataset Explorer (150 Samples)
                  </h3>
                  <p className="text-xs text-slate-500">
                    Search and filter records across all 4 morphological dimensions.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  {/* Search Input */}
                  <div className="relative">
                    <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
                    <input
                      type="text"
                      placeholder="Search ID or species..."
                      value={searchTerm}
                      onChange={(e) => {
                        setSearchTerm(e.target.value);
                        setCurrentPage(1);
                      }}
                      className="h-8 rounded-lg border border-slate-200 pl-8 pr-3 text-xs focus:border-blue-500 focus:outline-none"
                    />
                  </div>

                  {/* Species Filter */}
                  <select
                    value={speciesFilter}
                    onChange={(e) => {
                      setSpeciesFilter(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="h-8 rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-700 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="all">All Species (150)</option>
                    <option value="setosa">Setosa (50)</option>
                    <option value="versicolor">Versicolor (50)</option>
                    <option value="virginica">Virginica (50)</option>
                  </select>
                </div>
              </div>

              {/* Table */}
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-700">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-bold uppercase tracking-wider text-slate-600">
                      <th className="py-2.5 px-3">Sample ID</th>
                      <th className="py-2.5 px-3">Sepal Length (cm)</th>
                      <th className="py-2.5 px-3">Sepal Width (cm)</th>
                      <th className="py-2.5 px-3">Petal Length (cm)</th>
                      <th className="py-2.5 px-3">Petal Width (cm)</th>
                      <th className="py-2.5 px-3">Target Code</th>
                      <th className="py-2.5 px-3">Species</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {paginatedData.map((row) => (
                      <tr key={row.id} className="hover:bg-slate-50/60">
                        <td className="py-2 px-3 font-mono font-medium text-slate-500">#{row.id}</td>
                        <td className="py-2 px-3 font-mono">{row.sepalLength.toFixed(1)}</td>
                        <td className="py-2 px-3 font-mono">{row.sepalWidth.toFixed(1)}</td>
                        <td className="py-2 px-3 font-mono">{row.petalLength.toFixed(1)}</td>
                        <td className="py-2 px-3 font-mono">{row.petalWidth.toFixed(1)}</td>
                        <td className="py-2 px-3 font-mono font-bold text-slate-600">{row.target}</td>
                        <td className="py-2 px-3">
                          <span
                            className={`inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-semibold ${
                              row.species === "Setosa"
                                ? "bg-blue-50 text-blue-700"
                                : row.species === "Versicolor"
                                ? "bg-sky-50 text-sky-700"
                                : "bg-indigo-50 text-indigo-700"
                            }`}
                          >
                            {row.species}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination Bar */}
              <div className="mt-4 flex items-center justify-between border-t border-slate-100 pt-3 text-xs text-slate-500">
                <div>
                  Showing {filteredData.length > 0 ? (currentPage - 1) * pageSize + 1 : 0} to{" "}
                  {Math.min(currentPage * pageSize, filteredData.length)} of {filteredData.length} records
                </div>
                <div className="flex gap-1">
                  <button
                    disabled={currentPage === 1}
                    onClick={() => setCurrentPage((p) => p - 1)}
                    className="rounded border border-slate-200 bg-white px-2.5 py-1 font-medium disabled:opacity-40"
                  >
                    Previous
                  </button>
                  <span className="flex items-center px-2 font-mono">
                    {currentPage} / {totalPages || 1}
                  </span>
                  <button
                    disabled={currentPage >= totalPages}
                    onClick={() => setCurrentPage((p) => p + 1)}
                    className="rounded border border-slate-200 bg-white px-2.5 py-1 font-medium disabled:opacity-40"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ==================================================================== */}
        {/* TAB 3: PREPROCESSING */}
        {/* ==================================================================== */}
        {activeTab === "preprocessing" && (
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Data Preprocessing & Standardization
              </h2>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Pipeline, Missing Value Audit & Data Leakage Prevention
              </p>
            </div>

            {/* Connected Pipeline Flow */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Preprocessing Pipeline Flow
              </h3>
              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
                {[
                  { title: "Raw Data", sub: "150 Samples (4 Features)" },
                  { title: "Missing Audit", sub: "0 NaN / Null Values" },
                  { title: "Stratified Split", sub: "70% Train / 15% Val / 15% Test" },
                  { title: "StandardScaler", sub: "Fit ONLY on Train Set" },
                  { title: "One-Hot Encode", sub: "3-D Categorical Targets" },
                ].map((p, i) => (
                  <div key={p.title} className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-center">
                    <div className="text-xs font-bold text-slate-800">{p.title}</div>
                    <div className="mt-1 text-[11px] text-slate-500">{p.sub}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Split Breakdown */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <MetricCard
                title="Training Partition (70%)"
                value="104 Samples"
                subtitle="Used exclusively for backpropagation"
                highlight
              />
              <MetricCard
                title="Validation Partition (15%)"
                value="23 Samples"
                subtitle="Monitored for EarlyStopping & Val Loss"
              />
              <MetricCard
                title="Holdout Test Partition (15%)"
                value="23 Samples"
                subtitle="Completely unseen evaluation holdout"
              />
            </div>

            {/* Before vs After Scaling Table */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
                Before Scaling vs. After Scaling (First 4 Training Samples)
              </h3>
              <p className="text-xs text-slate-500">
                Demonstrating feature normalization into zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
              </p>

              <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                {/* Raw cm */}
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3.5">
                  <div className="text-xs font-bold text-slate-700">Raw Dimensions (Centimeters)</div>
                  <table className="mt-2 w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b text-[10px] uppercase text-slate-500">
                        <th className="py-1">Sepal L</th>
                        <th className="py-1">Sepal W</th>
                        <th className="py-1">Petal L</th>
                        <th className="py-1">Petal W</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {prepData.beforeScalingSample.map((row: number[], idx: number) => (
                        <tr key={`raw-${idx}`}>
                          {row.map((val: number, cIdx: number) => (
                            <td key={`rc-${idx}-${cIdx}`} className="py-1.5">
                              {val.toFixed(1)} cm
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Scaled z-scores */}
                <div className="rounded-xl border border-blue-200 bg-blue-50/50 p-3.5">
                  <div className="text-xs font-bold text-blue-800">Standardized (StandardScaler Z-Scores)</div>
                  <table className="mt-2 w-full text-left text-xs font-mono text-blue-900">
                    <thead>
                      <tr className="border-b border-blue-200 text-[10px] uppercase text-blue-600">
                        <th className="py-1">Sepal L (z)</th>
                        <th className="py-1">Sepal W (z)</th>
                        <th className="py-1">Petal L (z)</th>
                        <th className="py-1">Petal W (z)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-blue-100">
                      {prepData.afterScalingSample.map((row: number[], idx: number) => (
                        <tr key={`scaled-${idx}`}>
                          {row.map((val: number, cIdx: number) => (
                            <td key={`sc-${idx}-${cIdx}`} className="py-1.5">
                              {val.toFixed(3)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Data Leakage Callout */}
              <div className="mt-5 rounded-xl border border-blue-200 bg-blue-50/80 p-4 text-xs text-blue-900">
                <div className="font-bold">Crucial Academic Rule: Why Scaler is Fitted ONLY on Training Data</div>
                <p className="mt-1 leading-relaxed text-blue-800">
                  Fitting a scaler on the full dataset before splitting causes <b>data leakage</b>, because information
                  about the mean (&mu;) and standard deviation (&sigma;) of the unseen test set bleeds into the training
                  process. To maintain absolute scientific validity, the parameters (&mu;<sub>train</sub>, &sigma;<sub>train</sub>)
                  are computed strictly on the 104 training instances and then reused for transforming validation and test instances.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* ==================================================================== */}
        {/* TAB 4: ARCHITECTURE */}
        {/* ==================================================================== */}
        {activeTab === "architecture" && (
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Deep Feedforward Neural Network Architecture
              </h2>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                4 Fully Connected Layers &bull; 515 Trainable Parameters &bull; Adam + ReLU + Softmax
              </p>
            </div>

            {/* Interactive Architecture SVG Diagram & Parameter Table */}
            <ArchitectureDiagram />

            {/* Mathematical Foundations */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  1. Forward Propagation
                </h4>
                <div className="mt-2 text-lg font-bold font-mono text-blue-600">
                  z^[l] = W^[l] a^[l-1] + b^[l]
                </div>
                <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
                  Linear affine transformation where W<sup>[l]</sup> is the layer weight matrix, a<sup>[l-1]</sup> is the activation from
                  the previous layer, and b<sup>[l]</sup> is the bias vector.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  2. Hidden Activation (ReLU)
                </h4>
                <div className="mt-2 text-lg font-bold font-mono text-blue-600">
                  a^[l] = max(0, z^[l])
                </div>
                <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
                  Rectified Linear Unit mitigates the vanishing gradient problem, introduces non-linearity, and computes
                  gradients with high efficiency.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  3. Output Activation (Softmax)
                </h4>
                <div className="mt-2 text-lg font-bold font-mono text-blue-600">
                  P(y=i|x) = e^(z_i) / &Sigma; e^(z_j)
                </div>
                <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
                  Maps arbitrary real logits to a valid probability distribution over the 3 species such that &Sigma; P(y=i) = 1.0.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* ==================================================================== */}
        {/* TAB 5: TRAINING */}
        {/* ==================================================================== */}
        {activeTab === "training" && (
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Model Training & Optimization
              </h2>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Mini-Batch Gradient Descent with Adam & EarlyStopping (80 Epochs)
              </p>
            </div>

            {/* Offline vs Deployed Note */}
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-700">
              <span className="font-bold text-slate-900">Production Architecture Notice:</span> TensorFlow/Keras training
              is executed offline during the development build pipeline (via <code>training/train_model.py</code>). The trained
              515 weights and biases are exported to <code>public/model/model.json</code>. In production on Vercel, the web
              application runs ultra-fast forward-pass inference directly without retraining or requiring a serverless Python process.
            </div>

            {/* 3 Metrics */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <MetricCard
                title="Training Epochs"
                value={`${trainData.epochsTrained} Epochs`}
                subtitle="Restored optimal weights at epoch 68"
              />
              <MetricCard
                title="Training Accuracy"
                value={`${trainData.history.accuracy[trainData.history.accuracy.length - 1]}%`}
                subtitle="Converged smoothly to 100%"
                highlight
              />
              <MetricCard
                title="Validation Accuracy"
                value={`${trainData.history.val_accuracy[trainData.history.val_accuracy.length - 1]}%`}
                subtitle="Monitored for generalization"
                highlight
              />
            </div>

            {/* Interactive Training Curves */}
            <TrainingCurves
              history={trainData.history}
              earlyStoppingRestoredEpoch={trainData.earlyStoppingRestoredEpoch}
            />

            {/* Hyperparameters Card */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Optimization Hyperparameters & Objective
              </h3>
              <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4 text-xs">
                <div className="rounded-xl bg-slate-50 p-3">
                  <div className="text-slate-500">Optimizer</div>
                  <div className="mt-1 font-bold text-slate-900">Adam ($\beta_1=0.9, \beta_2=0.999$)</div>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <div className="text-slate-500">Learning Rate</div>
                  <div className="mt-1 font-bold font-mono text-slate-900">0.01</div>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <div className="text-slate-500">Batch Size</div>
                  <div className="mt-1 font-bold font-mono text-slate-900">16 samples / batch</div>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <div className="text-slate-500">EarlyStopping</div>
                  <div className="mt-1 font-bold text-slate-900">Patience = 15 Epochs</div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ==================================================================== */}
        {/* TAB 6: EVALUATION */}
        {/* ==================================================================== */}
        {activeTab === "evaluation" && (
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Independent Test Evaluation
              </h2>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Holdout Test Set Verification (23 completely unseen samples)
              </p>
            </div>

            {/* Prominent Test Accuracy Hero Card */}
            <div className="rounded-2xl border border-blue-200 bg-gradient-to-r from-blue-50/80 via-white to-blue-50/50 p-7 text-center shadow-sm">
              <div className="text-xs font-bold uppercase tracking-wider text-blue-700">
                Generalization Test Accuracy
              </div>
              <div className="mt-1 text-5xl font-black tracking-tight text-blue-600 sm:text-6xl">
                {evalData.testAccuracy}%
              </div>
              <p className="mt-2 text-xs text-slate-600">
                22 out of 23 test samples correctly classified &bull; Test Loss (Categorical Cross-Entropy):{" "}
                <span className="font-mono font-bold text-slate-900">{evalData.testLoss}</span>
              </p>
            </div>

            {/* Confusion Matrix & Classification Report */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <div className="lg:col-span-6">
                <ConfusionMatrix matrix={evalData.confusionMatrix} classNames={evalData.classNames} />
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-6">
                <div className="border-b border-slate-100 pb-3">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
                    Classification Report
                  </h3>
                  <p className="text-xs text-slate-500">
                    Precision, Recall, F1-Score, and Support per species.
                  </p>
                </div>

                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-700">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-bold uppercase tracking-wider text-slate-600">
                        <th className="py-2.5 px-3">Class</th>
                        <th className="py-2.5 px-3">Precision</th>
                        <th className="py-2.5 px-3">Recall</th>
                        <th className="py-2.5 px-3">F1-Score</th>
                        <th className="py-2.5 px-3 text-right">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono">
                      <tr>
                        <td className="py-2.5 px-3 font-sans font-semibold text-slate-900">Iris Setosa</td>
                        <td className="py-2.5 px-3 text-emerald-600 font-bold">1.00</td>
                        <td className="py-2.5 px-3 text-emerald-600 font-bold">1.00</td>
                        <td className="py-2.5 px-3 text-emerald-600 font-bold">1.00</td>
                        <td className="py-2.5 px-3 text-right">7</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 font-sans font-semibold text-slate-900">Iris Versicolor</td>
                        <td className="py-2.5 px-3 text-emerald-600 font-bold">1.00</td>
                        <td className="py-2.5 px-3">0.88</td>
                        <td className="py-2.5 px-3">0.93</td>
                        <td className="py-2.5 px-3 text-right">8</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 font-sans font-semibold text-slate-900">Iris Virginica</td>
                        <td className="py-2.5 px-3">0.89</td>
                        <td className="py-2.5 px-3 text-emerald-600 font-bold">1.00</td>
                        <td className="py-2.5 px-3">0.94</td>
                        <td className="py-2.5 px-3 text-right">8</td>
                      </tr>
                      <tr className="bg-slate-50/80 font-bold text-slate-900">
                        <td className="py-2.5 px-3 font-sans uppercase text-[11px]">Macro Average</td>
                        <td className="py-2.5 px-3">0.96</td>
                        <td className="py-2.5 px-3">0.96</td>
                        <td className="py-2.5 px-3">0.96</td>
                        <td className="py-2.5 px-3 text-right">23</td>
                      </tr>
                      <tr className="bg-blue-50/60 font-bold text-blue-900">
                        <td className="py-2.5 px-3 font-sans uppercase text-[11px]">Weighted Average</td>
                        <td className="py-2.5 px-3">0.96</td>
                        <td className="py-2.5 px-3">0.96</td>
                        <td className="py-2.5 px-3">0.96</td>
                        <td className="py-2.5 px-3 text-right">23</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Test Samples Actual vs Predicted Table */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700">
                Actual vs Predicted Test Sample Audit (23 Samples)
              </h3>
              <p className="text-xs text-slate-500">
                Detailed comparison for every individual holdout test sample.
              </p>

              <div className="mt-4 max-h-80 overflow-y-auto overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-700">
                  <thead className="sticky top-0 bg-slate-50">
                    <tr className="border-b border-slate-200 text-[11px] font-bold uppercase tracking-wider text-slate-600">
                      <th className="py-2.5 px-3">Sample</th>
                      <th className="py-2.5 px-3">Features [SL, SW, PL, PW]</th>
                      <th className="py-2.5 px-3">Actual Class</th>
                      <th className="py-2.5 px-3">Predicted Class</th>
                      <th className="py-2.5 px-3">Softmax Confidence</th>
                      <th className="py-2.5 px-3 text-right">Match</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {evalData.testSamples.map((sample: any) => (
                      <tr key={sample.sampleIndex} className="hover:bg-slate-50/60">
                        <td className="py-2 px-3 font-mono font-medium text-slate-500">
                          #{sample.sampleIndex}
                        </td>
                        <td className="py-2 px-3 font-mono text-[11px]">
                          [{sample.rawFeatures.join(", ")}]
                        </td>
                        <td className="py-2 px-3 font-medium text-slate-900">{sample.actualClass}</td>
                        <td className="py-2 px-3 font-medium text-blue-700">{sample.predictedClass}</td>
                        <td className="py-2 px-3 font-mono">{sample.confidence}%</td>
                        <td className="py-2 px-3 text-right">
                          {sample.match ? (
                            <span className="inline-flex items-center gap-1 rounded-md bg-emerald-50 px-2 py-0.5 text-[11px] font-bold text-emerald-700">
                              <CheckCircle2 className="h-3 w-3" />
                              Correct
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded-md bg-rose-50 px-2 py-0.5 text-[11px] font-bold text-rose-700">
                              <XCircle className="h-3 w-3" />
                              Misclassified
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {/* ==================================================================== */}
        {/* TAB 7: PREDICTION */}
        {/* ==================================================================== */}
        {activeTab === "predict" && (
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Interactive Prediction Laboratory
              </h2>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Real-Time Inference using the Trained 515-Parameter Deep FNN
              </p>
            </div>

            {/* Real Predictor Component */}
            <Predictor model={modelData} />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-14 border-t border-slate-200 pt-6 text-center text-xs text-slate-500">
        <p className="font-semibold text-slate-700">
          Amogh Samarth &bull; USN: CM23034
        </p>
        <p className="mt-1">
          Pattern Recognition &bull; TAE 1: Project Based Learning &bull; Phase I &bull; Ready for Vercel Deployment
        </p>
      </footer>
    </div>
  );
}
