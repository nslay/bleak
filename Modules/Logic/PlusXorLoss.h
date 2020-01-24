#pragma once

#ifndef BLEAK_PLUSXORLOSS_H
#define BLEAK_PLUSXORLOSS_H

#include "Vertex.h"
#include "LogicOperation.h"

namespace bleak {

template<typename RealType>
class PlusXorLoss : public Vertex<RealType> {
public:
  bleakNewVertex(PlusXorLoss, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inLabels"),
    bleakAddOutput("outLoss"),
    bleakAddOutput("work"),
    bleakAddProperty("penaltyWeight", m_fPenaltyWeight));

  bleakForwardVertexTypedefs();

  virtual ~PlusXorLoss() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInLabels, "inLabels", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);
    bleakGetAndCheckOutput(p_clWork, "work", false);

    if (p_clWork->HasTargets()) {
      std::cerr << GetName() << ": Error: The 'work' output should have no targets." << std::endl;
      return false;
    }

    const ArrayType &clInLabels = p_clInLabels->GetData();
    const ArrayType &clInData = p_clInData->GetData();

    if (clInLabels.GetSize().GetDimension() != clInData.GetSize().GetDimension()) {
      std::cerr << GetName() << ": Error: Incompatible dimensions: inLabels = " << clInLabels.GetSize().GetDimension() << ", inData = " << clInData.GetSize().GetDimension() << std::endl;
      return false;
    }

    if (clInLabels.GetSize()[0] != clInData.GetSize()[0] || clInLabels.GetSize().SubSize(1) != clInData.GetSize().SubSize(1)) {
      std::cerr << GetName() << ": Error: Incompatible sizes: inLabels = " << clInLabels.GetSize() << ", inData = " << clInData.GetSize() << std::endl;
      return false;
    }

    ArrayType &clOutLoss = p_clOutLoss->GetData();
    ArrayType &clOutGradient = p_clOutLoss->GetGradient();
    ArrayType &clWork = p_clWork->GetData();
    ArrayType &clWorkGradient = p_clWork->GetGradient();

    const Size clSize = { 1 };    

    clOutLoss.SetSize(clSize);
    clOutGradient.SetSize(clSize);

    clWork.SetSize(clInData.GetSize());
    clWorkGradient.SetSize(clInData.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

  virtual void Forward() override {
    const PlusOrOp<RealType> clOr;
    const PlusXorOp<RealType> clXor;

    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInLabels, "inLabels");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();

    const int iCount = clInData.GetSize().Count();

    const RealType * const inData = clInData.data();
    const RealType * const inLabels = clInLabels.data();

    RealType &outLoss = *p_clOutLoss->GetData().data();

    outLoss = clOr.Identity();

    for (int i = 0; i < iCount; ++i)
      outLoss = clOr(outLoss, clXor(inData[i], inLabels[i]));

    outLoss *= RealType(m_fPenaltyWeight) / clInData.GetSize()[0];
  }

  virtual void Backward() override {
    const PlusOrOp<RealType> clOr;
    const PlusXorOp<RealType> clXor;

    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInLabels, "inLabels");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");
    bleakGetAndCheckOutput(p_clWork, "work");

    if (!p_clInData->GetGradient().Valid()) // Nothing to do!
      return;

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();
    ArrayType &clInGradient = p_clInData->GetGradient();
    ArrayType &clWork = p_clWork->GetData();
    ArrayType &clWorkGradient = p_clWork->GetGradient();

    const int iCount = clInData.GetSize().Count();

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inLabels = clInLabels.data();

    RealType * const p_inGradient = clInGradient.data();
    RealType * const p_work = clWork.data();
    RealType * const p_workGradient = clWorkGradient.data();

    RealType &outGradient = *p_clOutLoss->GetGradient().data();

    if (IsLeaf())
      outGradient = RealType(1);

    // Fill the work variable
    p_work[0] = clOr.Identity();
    for (int i = 1; i < iCount; ++i)
      p_work[i] = clOr(p_work[i-1], clXor(p_inData[i-1], p_inLabels[i-1]));

    p_workGradient[iCount-1] = RealType(1);
    for (int i = iCount-2; i >= 0; --i)
      p_workGradient[i] = p_workGradient[i+1] * clOr.DiffA(p_work[i+1], clXor(p_inData[i+1], p_inLabels[i+1]));

    const RealType scale = RealType(m_fPenaltyWeight) * outGradient / clInData.GetSize()[0];

    for (int i = 0; i < iCount; ++i) {
      p_inGradient[i] += scale * p_workGradient[i] * clOr.DiffB(p_work[i], clXor(p_inData[i], p_inLabels[i])) * clXor.DiffA(p_inData[i], p_inLabels[i]);
    }
  }

protected:
  PlusXorLoss() = default;

private:
  float m_fPenaltyWeight = 1.0f;
};

} // end namespace bleak

#endif // !BLEAK_PLUSXORLOSS_H
