defmodule Bumblebee.Text.EsmTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  describe "integration" do
    test "encoding model input" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/esm2_t6_8M_UR50D"})

      assert %Bumblebee.Text.EsmTokenizer{} = tokenizer

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, [
          "LAGVS",
          "WCB"
          # {"LAGVS", "WCB"}
        ])

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([
          [0, 4, 5, 6, 7, 8, 2], 
          [0, 22, 23, 25, 2]
        ])
      )

      # assert_equal(
      #   inputs["attention_mask"],
      #   Nx.tensor([
      #     [1, 1, 1, 1, 1, 1, 1, 1, 1],
      #     [1, 1, 1, 1, 1, 1, 1, 0, 0]
      #   ])
      # )
      #
      # assert_equal(
      #   inputs["token_type_ids"],
      #   Nx.tensor([
      #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
      #     [0, 0, 0, 0, 0, 0, 0, 0, 0]
      #   ])
      # )
    end
  end
end
